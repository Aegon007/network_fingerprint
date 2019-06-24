#!/usr/bin/python

import os
import sys
import time

sys.path.append('bin')
import nFoldCrossValidation
import draw_graph

import utils


class MyOptions():
    def __init__(self, model, dataDir, nFold, itv, word2vec=0, file2store=0):
        self.model = model
        self.dataDir = dataDir
        self.nFold = nFold
        self.word2vec = word2vec
        self.file2store = file2store
        self.interval = itv


def main():
    datapath = '/home/carl/work_dir/echo_proj_phase_2/data/gamma_100/gamma'

    method_itv_pair = {'Bayes':1000, 'VNGpp':10000, 'Cumul':0, 'Adaboost':10000, 'Svm':10000}

    tmpList = []
    for method in method_itv_pair.keys(): 
        print('start test with method {}'.format(method))
        itv_val = method_itv_pair[method]
        opts = MyOptions(method, datapath, 5, itv_val)
        start = time.time()
        nFoldCrossValidation.main(opts)
        end = time.time()
        tmpLine = 'test with method {} using time {}'.format(method, str(end - start))
        tmpList.append(tmpLine)

    with open('time_test.res', 'w') as f:
        content = '\n'.join(tmpList)
        content = content + '\n'
        f.write(content)


if __name__ == "__main__":
    main()
