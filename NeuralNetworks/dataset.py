import numpy as np
import gzip as gz
import pickle as pkl


def loadDataset():
    f = gz.open('dataset.pkl.gz', 'rb')
    tra, val = pkl.load(f, encoding='latin1')
    f.close()
    tra = {'X': tra[0],
           'Y': tra[1]}
    val = {'X': val[0],
           'Y': val[1]}
    
    return tra, val
