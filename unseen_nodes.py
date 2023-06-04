import numpy as np

class SpatialSplit:
    def __init__(self, length, r_trn = .7, r_val=.1, r_tst=.2, seed=0):
        # trn+val+tst might be < 1, to do further analysis
        # is the sum > 1, then it is sampled with replacement (to simulate all nodes seen)
        self.length = length
        self.r_trn = r_trn
        self.r_trnval = self.r_trn + r_val
        self.r_val = r_val
        self.r_tst = r_tst
        np.random.seed(seed=seed) # reset the RNG
        if r_trn + r_val + r_tst<1.0:
            # no replacement
            indices = np.arange(length)
            np.random.shuffle(indices)
            i_split1 = int(length*(r_trn))
            i_split2 = int(length*(r_trn+r_val))
            i_split3 = int(length*(r_trn+r_val+r_tst))
            self.i_trn = indices[        :i_split1]
            self.i_val = indices[i_split1:i_split2]
            self.i_tst = indices[i_split2:i_split3]
        else:
            # with replacement
            self.i_trn = np.random.choice(length, size=int(length*(r_trn)), replace=False)
            self.i_val = np.random.choice(length, size=int(length*(r_val)), replace=False)
            self.i_tst = np.random.choice(length, size=int(length*(r_tst)), replace=False)

    def __repr__(self):
        return 'all: trn/val/tst : ' +\
            str(self.length)+' : '+\
            str(len(self.i_trn))+' / '+\
            str(len(self.i_val))+' / '+\
            str(len(self.i_tst))
