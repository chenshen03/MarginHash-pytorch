import os
import os.path as osp
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import datasets
from utils import *


parser = argparse.ArgumentParser("Center Loss")
parser.add_argument('-d', '--dataset', type=str, default='cifar-s1')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--save-dir', type=str, default='./snapshot/Hash/cifar-s1')
parser.add_argument('--prefix', type=str, default='q0.01')
parser.add_argument('--tencrop', action='store_true')

args = parser.parse_args()
print(args)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True

    print("Creating dataset: {}".format(args.dataset))
    dataset = datasets.CIFARS1(args.batch_size, use_gpu, num_workers=4, tencrop=args.tencrop)
    testloader, databaseloader = dataset.testloader, dataset.databaseloader

    print("load pretrained model: {}".format(args.prefix))
    model_path = osp.join(osp.join(args.save_dir, args.prefix), 'model_best_mAP.pth')
    model = torch.load(model_path)

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    print("==> Test")
    acc = test(model, testloader, args.tencrop, use_gpu)
    print(f'acc:{acc}')

    print("==> Evaluate")
    mAP_feat, mAP_sign, _ = evaluate(model, databaseloader, testloader, dataset.R, args.tencrop, use_gpu)
    print(f'mAP_feat:{mAP_feat:.4f}  mAP_sign:{mAP_sign:.4f}')


def test(model, testloader, tencrop, use_gpu):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()

            if tencrop:
                bs, ncrops, c, h, w = data.size()
                _, outputs = model(data.view(-1, c, h, w))
                outputs = outputs.view(bs, ncrops, -1).mean(1)
            else:
                _, outputs = model(data)
            
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

    acc = correct * 100. / total
    return acc


def evaluate(model, dbloader, testloader, R, tencrop, use_gpu):
    model.eval()

    print('calculate database codes...')
    db_feats = []
    db_labels = []
    with torch.no_grad():
        for data, labels in dbloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()

            if tencrop:
                bs, ncrops, c, h, w = data.size()
                feats, _ = model(data.view(-1, c, h, w))
                feats = feats.view(bs, ncrops, -1).mean(1)
            else:
                feats, _ = model(data)
            
            db_feats.append(feats.data.cpu().numpy())
            db_labels.append(labels.data.cpu().numpy())
    db_feats = np.concatenate(db_feats, 0)
    db_codes = sign(db_feats)
    db_labels = np.concatenate(db_labels, 0)
    db_labels = np.eye(10)[db_labels]

    print('calculate test codes...')
    test_feats = []
    test_labels = []
    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()

            if tencrop:
                bs, ncrops, c, h, w = data.size()
                feats, _ = model(data.view(-1, c, h, w))
                feats = feats.view(bs, ncrops, -1).mean(1)
            else:
                feats, _ = model(data)

            test_feats.append(feats.data.cpu().numpy())
            test_labels.append(labels.data.cpu().numpy())
    test_feats = np.concatenate(test_feats, 0)
    test_codes = sign(test_feats)
    test_labels = np.concatenate(test_labels, 0)
    test_labels = np.eye(10)[test_labels]

    print('calculate mAP...')
    mAP_feat = get_mAP(db_feats, db_labels, test_feats, test_labels, R)
    mAP_sign = get_mAP(db_codes, db_labels, test_codes, test_labels, R)
    
    code_and_labels = {'db_feats':db_feats, 'db_codes':db_codes, 'db_labels':db_labels,
                       'test_feats': test_feats, 'test_codes':test_codes, 'test_labels':test_labels}
    return mAP_feat, mAP_sign, code_and_labels


if __name__ == "__main__":
    main()