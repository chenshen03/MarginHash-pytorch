import os
import os.path as osp
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import datasets
from utils import *


parser = argparse.ArgumentParser("Hash Test")
parser.add_argument('--dataset', type=str, default='cifar_s1')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--save-dir', type=str, default='./snapshot/cifar_s1')
parser.add_argument('--prefix', type=str, default='debug')
parser.add_argument('--tencrop', action='store_true')
parser.add_argument('--plot', action='store_true')

args = parser.parse_args()
args.save_dir = f'snapshot/{args.dataset}/{args.prefix}'
print(args)


def main():
    sys.stdout = Logger(osp.join(args.save_dir, 'test.log'))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print("Currently using GPU: {}".format(args.gpus))
        cudnn.benchmark = True

    print("Creating dataset: {}".format(args.dataset))
    dataset = datasets.create(name=args.dataset, batch_size=args.batch_size, bit=32, tencrop=args.tencrop)
    testloader, databaseloader = dataset.testloader, dataset.databaseloader

    model_path = osp.join(args.save_dir, 'model_best.pth')
    print("load pretrained model: {}".format(model_path))
    model = torch.load(model_path)
    if multi_gpus:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    print("==> Evaluate")
    mAP_feat, mAP_sign, mAP_topK, code_and_labels = evaluate(model, databaseloader, testloader, dataset.R, args.tencrop, device)
    np.save(osp.join(args.save_dir, 'code_and_label.npy'), code_and_labels)
    print(f'mAP_feat:{mAP_feat:.4f}  mAP_sign:{mAP_sign:.4f}  mAP_top1000:{mAP_topK:.4f}')


def evaluate(model, databaseloader, testloader, R, tencrop, device):
    model.eval()

    print('calculate database codes...')
    db_feats = []
    db_labels = []
    with torch.no_grad():
        for data, labels in databaseloader:
            data, labels = data.to(device), labels.to(device)

            if tencrop:
                bs, ncrops, c, h, w = data.size()
                features = model(data.view(-1, c, h, w))
                features = features.view(bs, ncrops, -1).mean(1)
            else:
                features = model(data)

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

            if tencrop:
                bs, ncrops, c, h, w = data.size()
                features = model(data.view(-1, c, h, w))
                features = features.view(bs, ncrops, -1).mean(1)
            else:
                features = model(data)

            test_feats.append(features.data.cpu().numpy())
            test_labels.append(labels.data.cpu().numpy())
    test_feats = np.concatenate(test_feats, 0)
    test_codes = sign(test_feats)
    test_labels = np.concatenate(test_labels, 0)

    print('calculate mAP...')
    mAP_feat = get_mAP(db_feats, db_labels, test_feats, test_labels, R)
    mAP_sign = get_mAP(db_codes, db_labels, test_codes, test_labels, R)
    mAP_topK = get_mAP(db_codes, db_labels, test_codes, test_labels, R=1000)
    
    code_and_labels = {'db_feats':db_feats, 'db_codes':db_codes, 'db_labels':db_labels,
                       'test_feats': test_feats, 'test_codes':test_codes, 'test_labels':test_labels}
    if args.plot:
        plot_features(db_feats, db_labels.argmax(1), num_classes=10, epoch=1, save_dir=args.save_dir, prefix='database')
        plot_features(test_feats, test_labels.argmax(1), num_classes=10, epoch=1, save_dir=args.save_dir, prefix='test')

    return mAP_feat, mAP_sign, mAP_topK, code_and_labels


if __name__ == "__main__":
    main()