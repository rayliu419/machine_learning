#encoding=utf-8

import time

import numpy as np

if __name__ == '__main__':
    vec_1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    vec_2 = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,]
    vec_1 = np.array(vec_1)
    vec_2 = np.array(vec_2)
    print vec_1 - vec_2
    print np.square(vec_1 - vec_2)
    print np.sqrt(np.sum(np.square(vec_1 - vec_2)))
    print np.linalg.norm(vec_1 - vec_2)