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
parser.add_argument('--dataset', type=str, default='cifar-s1')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--save-dir', type=str, default='./snapshot/Hash/cifar-s1/debug')
parser.add_argument('--tencrop', action='store_true')

args = parser.parse_args()
print(args)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print("Currently using GPU: {}".format(args.gpus))
        cudnn.benchmark = True

    print("Creating dataset: {}".format(args.dataset))
    dataset = datasets.CIFARS1(args.batch_size, tencrop=args.tencrop)
    testloader, databaseloader = dataset.testloader, dataset.databaseloader

    print("load pretrained model: {}".format(args.save_dir))
    model_path = osp.join(args.save_dir, 'model_best.pth')
    model = torch.load(model_path)
    if multi_gpus:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    print("==> Evaluate")
    mAP_feat, mAP_sign, _ = evaluate(model, databaseloader, testloader, dataset.R, args.tencrop, device)
    print(f'mAP_feat:{mAP_feat:.4f}  mAP_sign:{mAP_sign:.4f}')


def evaluate(model, databaseloder, testloader, R, tencrop, device):
    model.eval()

    print('calculate database codes...')
    db_feats = []
    db_labels = []
    with torch.no_grad():
        for data, labels in databaseloder:
            data, labels = data.to(device), labels.to(device)

            if tencrop:
                bs, ncrops, c, h, w = data.size()
                feats = model(data.view(-1, c, h, w))
                feats = feats.view(bs, ncrops, -1).mean(1)
            else:
                feats = model(data)[0]

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
            data, labels = data.to(device), labels.to(device)

            if tencrop:
                bs, ncrops, c, h, w = data.size()
                feats = model(data.view(-1, c, h, w))
                feats = feats.view(bs, ncrops, -1).mean(1)
            else:
                feats = model(data)[0]

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