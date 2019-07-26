import numpy as np
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
