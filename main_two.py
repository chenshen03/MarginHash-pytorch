import os
import sys
import argparse
import datetime
import time
import os.path as osp
import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import datasets
import backbone
import margin
from utils import *
from loss import *


parser = argparse.ArgumentParser("Hash Train")
# dataset
parser.add_argument('--dataset', type=str, default='cifar_s1')
# optimization
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=200)
parser.add_argument('--stepsize', type=int, default=50)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
# mode
parser.add_argument('--feat-dim', type=int, default=32)
parser.add_argument('--backbone', type=str, default='AlexNet')
parser.add_argument('--classifier', type=str, default='Softmax', help='Softmax, ArcFace, CosFace, SphereFace, MultiMargin')
parser.add_argument('--scale', type=float, default=32.0, help='scale size')
# misc
parser.add_argument('--print-freq', type=int, default=10)
parser.add_argument('--test-freq', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=50)
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--save-dir', type=str, default='snapshot/cifar_s1')
parser.add_argument('--prefix', type=str, default='debug')
parser.add_argument('--plot', action='store_true', help="whether to plot features for every epoch")

args = parser.parse_args()
args.save_dir = f'snapshot/{args.dataset}/{args.prefix}'
print(args)


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) #cpu
    torch.cuda.manual_seed_all(seed)  #并行gpu
    torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
    torch.backends.cudnn.benchmark = True   #训练集变化不大时使训练加速


def main():
    setup_seed(args.seed)
    sys.stdout = Logger(osp.join(args.save_dir, 'log_' + args.dataset + '.txt'))
    # gpu init
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print("Currently using GPU: {}".format(args.gpus))
    else:
        print("Currently using CPU")

    # define dataset, backbone and margin layer
    print("Creating dataset: {}".format(args.dataset))
    dataset = datasets.create(name=args.dataset, batch_size=args.batch_size)
    trainloader, testloader, databaseloader = dataset.trainloader, dataset.testloader, dataset.databaseloader

    print("Creating net: {}".format(args.backbone))
    net = backbone.create(name=args.backbone, feat_dim=args.feat_dim)

    print("Creating classifier: {}".format(args.classifier))
    classifier = margin.create(name=args.classifier, feat_dim=4096, 
                               num_classes=dataset.num_classes, scale=args.scale)

    # define optimizers for different layer
    criterion_xent = nn.CrossEntropyLoss()
    parameter_list = [{"params":net.feature_layers.parameters(), "lr":args.lr}, \
                      {"params":net.hash_layer.parameters(), "lr":args.lr}, \
                      {"params":classifier.parameters(), "lr":args.lr}]
    optimizer_model = torch.optim.SGD(parameter_list, lr=args.lr, weight_decay=5e-04, momentum=0.9)

    if args.stepsize > 0:
        scheduler = lr_scheduler.StepLR(optimizer_model, step_size=args.stepsize, gamma=args.gamma)

    if multi_gpus:
        net = nn.DataParallel(net).to(device)
        classifier = nn.DataParallel(classifier).to(device)
    else:
        net = net.to(device)
        classifier = classifier.to(device)

    start_time = time.time()

    best_acc, best_mAP_feat, best_mAP_sign= 0.0, 0.0, 0.0
    for epoch in range(args.max_epoch):
        print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
        train(net, classifier,
              criterion_xent, optimizer_model,
              trainloader, dataset, epoch, device)

        if args.stepsize > 0: scheduler.step()

        if args.test_freq > 0 and (epoch+1) % args.test_freq == 0 or (epoch+1) == args.max_epoch:
            print("==> Test")
            acc, err = test(net, classifier, testloader, device)
            print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))
            if acc > best_acc:
                best_acc = acc

        if args.eval_freq > 0 and (epoch+1) % args.eval_freq == 0 or (epoch+1) == args.max_epoch:
            print("==> Evaluate")
            mAP_feat, mAP_sign, code_and_labels = evaluate(net, databaseloader, testloader, \
                                                dataset.R, dataset.num_classes, epoch, device)            
            print(f'mAP_feat:{mAP_feat:.4f}  mAP_sign:{mAP_sign:.4f}')
            if mAP_sign > best_mAP_sign:
                best_mAP_sign = mAP_sign
                best_mAP_feat = mAP_feat
                print(f'best mAP updated to {best_mAP_sign:.4f}')
                np.save(osp.join(args.save_dir, 'code_and_label.npy'), code_and_labels)
                torch.save(net,  osp.join(args.save_dir, 'model_best.pth'))

    print(f"best mAP_feat:{best_mAP_feat:.4f}  best mAP_sign:{best_mAP_sign:.4f}  best Acc:{best_acc}")
    torch.save(net,  osp.join(args.save_dir, 'model_final.pth'))
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train(net, classifier, 
          criterion_xent, optimizer_model,
          trainloader, dataset, epoch, device):
    net.train()
    losses = AverageMeter()
    c_losses = AverageMeter()
    s_losses = AverageMeter()
    
    if args.plot:
        all_features, all_labels = [], []

    for batch_idx, (data, labels) in enumerate(trainloader):
        labels_onehot = labels.to(device)
        data, labels = data.to(device), labels.argmax(1).to(device)\
        
        # compute output
        features, fc1 = net(data)
        outputs = classifier(fc1, labels)
        c_loss = criterion_xent(outputs, labels)
        s_loss = hadamard_loss(features, labels_onehot, dataset.hadamard)
        q_loss = quantization_loss(features)
        b_loss = balance_loss(features)
        i_loss = independence_loss(features)
        loss = 3 * c_loss + 1 * s_loss + 0.01 * q_loss + 0 * b_loss + 0 * i_loss

        # compute gradient and do SGD step
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()
        
        losses.update(loss.item(), labels.size(0))
        c_losses.update(c_loss.item(), labels.size(0))
        s_losses.update(s_loss.item(), labels.size(0))

        if args.plot:
            all_features.append(features.data.cpu().numpy())
            all_labels.append(labels.data.cpu().numpy())

        if (batch_idx+1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}) c_loss {:.6f} ({:.6f}) s_Loss {:.6f} ({:.6f})" \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg, c_losses.val, c_losses.avg, s_losses.val, s_losses.avg))

    if args.plot:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        plot_features(all_features, all_labels, dataset.num_classes, epoch, save_dir=args.save_dir, prefix='train')


def test(net, classifier, testloader, device):
    net.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.to(device), labels.argmax(1).to(device)
            # compute output
            features, fc1 = net(data)
            outputs = classifier(fc1, labels)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()
            
    acc = correct * 100. / total
    err = 100. - acc
    return acc, err


def evaluate(net, databaseloader, testloader, R, num_classes, epoch, device):
    net.eval()

    print('calculate database codes...')
    db_feats = []
    db_labels = []
    with torch.no_grad():
        for data, labels in databaseloader:
            data, labels = data.to(device), labels.to(device)
            features, _ = net(data)
            db_feats.append(features.data.cpu().numpy())
            db_labels.append(labels.data.cpu().numpy())
    db_feats = np.concatenate(db_feats, 0)
    db_codes = sign(db_feats)
    db_labels = np.concatenate(db_labels, 0)

    print('calculate test codes...')
    test_feats = []
    test_labels = []
    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.to(device), labels.to(device)
            features, _ = net(data)
            test_feats.append(features.data.cpu().numpy())
            test_labels.append(labels.data.cpu().numpy())
    test_feats = np.concatenate(test_feats, 0)
    test_codes = sign(test_feats)
    test_labels = np.concatenate(test_labels, 0)

    print('calculate mAP...')
    mAP_feat = get_mAP(db_feats, db_labels, test_feats, test_labels, R)
    mAP_sign = get_mAP(db_codes, db_labels, test_codes, test_labels, R)
    
    code_and_labels = {'db_feats':db_feats, 'db_codes':db_codes, 'db_labels':db_labels,
                       'test_feats': test_feats, 'test_codes':test_codes, 'test_labels':test_labels}
    if args.plot:
        plot_features(db_feats, db_labels.argmax(1), num_classes, epoch, save_dir=args.save_dir, prefix='database')
        plot_features(test_feats, test_labels.argmax(1), num_classes, epoch, save_dir=args.save_dir, prefix='test')

    return mAP_feat, mAP_sign, code_and_labels


if __name__ == '__main__':
    main()
