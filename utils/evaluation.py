import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from .distance import distance


def get_mAP(db_codes, db_labels, test_codes, test_labels, R):
    query_num = test_codes.shape[0]
    sim = np.dot(db_codes, test_codes.T)
    ids = np.argsort(-sim, axis=0)
    APx = []
    labels = np.copy(test_labels)
    for i in range(query_num):
        label = labels[i, :]
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(db_labels[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R+1, 1)
        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
    mAP = np.mean(np.array(APx)) if len(APx) > 0 else 0
    return mAP


def get_RAMAP(db_output, db_labels, q_output, q_labels, cost=False):
    ''' 
    - On the Evaluation Metric for Hashing
    '''
    M, Q = q_output.shape
    R = Q
    RAAPs = []
    time_costs = [comb(Q, r) for r in range(Q+1)]
    distH = distance(q_output, db_output, pair=False, dist_type='hamming')
    gnds = np.dot(q_labels, db_labels.transpose()) > 0
    for i in range(M):
        gnd = gnds[i,:]
        hamm = distH[i,:]
        RAAP = 0
        for r in range(R+1):
            hamm_r_idx = np.where(hamm<=r)
            rel = len(hamm_r_idx[0])
            if(rel == 0):
                continue
            imatch = np.sum(gnd[hamm_r_idx])
            if cost:
                time_cost = np.sum(time_costs[:r+1])
                RAAP += (imatch / (rel * time_cost))
            else:
                RAAP += (imatch / rel)
        RAAP = RAAP / (R + 1)
        RAAPs.append(RAAP)
    return np.mean(RAAPs)


def get_precision_top(db_codes, db_labels, test_codes, test_labels, k=500):
    dist = distance(test_codes, db_codes, dist_type='hamming', pair=False)
    index = np.argsort(dist, axis=1)
    q_num = len(test_codes)
    precision = 0
    for i in range(q_num):
        precision += (np.sum(db_labels[index[i][:k]] == test_labels[i]) / k)
    precision /= q_num
    return precision


def get_pre_recall(q_feats, q_labels, db_feats, db_labels):
    eps = 1e-6
    dist = distance(q_feats, db_feats, dist_type='hamming', pair=False)
    S = np.matmul(q_labels, db_labels.transpose())
    S[S==-1] = 0
    total_good_pairs = np.sum(S)
    max_hamming = int(np.maximum(np.max(dist), 3))
    precision = np.zeros(max_hamming+1)
    recall = np.zeros(max_hamming+1)
    for i in range(max_hamming+1):
        index = dist <= (i + eps)
        retrieved_good_pairs = np.sum(S[index])
        retrieved_pairs = np.sum(index)
        precision[i] = retrieved_good_pairs / (retrieved_pairs + eps)
        recall[i] = retrieved_good_pairs / total_good_pairs
    return precision, recall


def save_pre_recall(precision, recall, path):
    plt.figure()
    plt.plot(precision, recall)
    plt.title('Precision with Recall Curve')
    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.savefig(os.path.join(path, 'pre_recall_curve.jpg'))
    np.save(os.path.join(path, 'pre_recall.npy'), {'precision':precision, 'recall':recall})
