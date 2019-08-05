import os
import sys
import errno
import shutil
import os.path as osp
import numpy as np
import torch
from scipy.linalg import hadamard


def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))


class Logger(object):
    """
    Write console output to external text file.
    
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def sign(x):
    s = np.sign(x)
    tmp = s[s == 0]
    s[s==0] = np.random.choice([-1, 1], tmp.shape)
    return s


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        # self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.data = next(self.loader)
        except StopIteration:
            self.next_input = None
            return
        # with torch.cuda.stream(self.stream):
        #     self.next_data = self.next_data.cuda(non_blocking=True)
            
    def next(self):
        # torch.cuda.current_stream().wait_stream(self.stream)
        data = self.data
        self.preload()
        return data

    
def generate_hadamard_codebook(bit, num_classes):
    
    def balance_loss(H):
        H_mean = np.mean(H, axis=0)
        loss = np.mean(H_mean ** 2)
        return loss

    def independence_loss(H):
        batch_size, bit = H.shape
        I = np.eye(batch_size)
        loss = np.mean(((H @ H.transpose()) / bit - I) ** 2)
        return loss
    
    if bit < num_classes:
        for i in [16, 32, 64, 128, 256]:
            if i > num_classes: num_hadamard = i; break
    else:
        num_hadamard = bit
    if num_hadamard == 48:
        HC_origin = np.loadtxt('../data/hadamard_codebook/hadamard_48bit.txt')
    else:
        HC_origin = hadamard(num_hadamard)
    
    bl_min = 10000
    hadamard_codebook = None
    for i in range(10000):
        if bit < num_classes:
            lshW = np.random.randn(num_hadamard, bit)
            # lshW = lshW / np.tile(np.diag(np.sqrt(lshW.transpose() @ lshW)), (num_hadamard, 1))
            HC = np.sign(HC_origin @ lshW)
        else:
            HC = HC_origin
        HC = HC[np.random.permutation(num_hadamard), :]
        HC_tmp = HC[:num_classes, :]
        bl = balance_loss(HC_tmp)
        if bl < bl_min:
            bl_min = bl
            hadamard_codebook = HC_tmp
    return hadamard_codebook
