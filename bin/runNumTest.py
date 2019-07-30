#!/usr/bin/python

import os
import sys
import argparse
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import numpy as np
from collections import defaultdict
import time

import trainByBayes
import trainByJaccard
import trainByVNGpp
import trainBySvm
import trainByAdaboost
import trainByCumul

import testByBayes
import testByJaccard
import testByVNGpp
import testBySvm
import testByAdaboost
import testByCumul
import prepareDataNum

import utils
import fileUtils
import parseWord2VecFile


NUM_RES_DIR = 'num_test_res'
if not os.path.isdir(NUM_RES_DIR):
    os.makedirs(NUM_RES_DIR)


def write2file(contents, fpath):
    output = os.path.join(NUM_RES_DIR, fpath)
    with open(output, 'w') as f:
        f.write(contents)


class MyOptions():
    def __init__(self, input, output):
        self.model = ''
        self.input = input
        self.output = output
        self.interval = 0

    def setModel(self, model):
        # set interval value based model name
        self.model = model
        if 'Adaboost' == model:
            self.interval = 10000


def oneRun(modelName, X_train, Y_train, X_test, labelMap):
    if 'Adaboost' == modelName:
        model = trainByAdaboost.train(X_train, Y_train)
        predictions = testByAdaboost.test(model, X_test)
        #str_predictions = nFord.convert2Str(predictions, labelMap)
    elif 'Cumul' == modelName:
        context = trainByCumul.generateContext()
        model = trainByCumul.train(X_train, Y_train, context)
        predictions = testByCumul.test(model, X_test)
        #str_predictions = nFord.convert2Str(predictions, labelMap)
    else:
        raise ValueError('for now we only do this test for Adaboost and Cumul')

    return predictions


def oneTest(testNums, opts):
    content_list = []
    for num in testNums:
        print("start extracting feature for num {:d}".format(num))
        allData, allLabel, labelMap = prepareDataNum.loadData(opts, num)
        newData = preprocessing.scale(allData)

        X_train, X_test, Y_train, Y_test = train_test_split(newData, allLabel, test_size=0.2, shuffle=True, random_state=77)
        print('now start training...')
        predictions = oneRun(opts.model, X_train, Y_train, X_test, labelMap)
        acc = accuracy_score(Y_test, predictions)
        content = 'model {} with sample number {:d} has acc {:f}'.format(opts.model, num, acc)
        content_list.append(content)

    allContents = '\n#######################\n'.join(content_list)
    allContents = allContents + '\n'
    print(allContents)
    write2file(allContents, opts.output)


def main(opts):
    testModels = ['Adaboost', 'Cumul']
    testNums = list(range(100, 1400, 100))
    print(testNums)
    print('now start num test for given models {}'.format(testModels))
    opts = MyOptions(opts.input, opts.output)
    for model in testModels:
        print("selecting model {} for testing...".format(opts.model))
        opts.setModel(model)
        oneTest(testNums, opts)


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='data dir where store all data')
    parser.add_argument('-o', '--output', help='file store result data')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
