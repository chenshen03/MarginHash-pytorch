import numpy as np


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
    
    return np.mean(np.array(APx))