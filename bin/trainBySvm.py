#!/usr/bin/python

import os
import sys
import argparse
from collections import defaultdict
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import numpy as np

import fileUtils
import tools
import trainByVNGpp
import trainByJaccard
from trainByBayes import saveModel
import nFoldCrossValidation
import testBySvm


def computeFeature(fpath, rangeList):
    upPackNum, downPackNum, upStreamTotal, downStreamTotal, traceTimeList, tupleList = trainByVNGpp.readfile(fpath)
    # burst bytes
    burstList = trainByVNGpp.calculateBursts(tupleList)
    start, end, interval = rangeList[0], rangeList[1], rangeList[2]
    rangeList, sectionList = tools.getSectionList(start, end, interval)
    for feat in burstList:
        index = tools.computeRange(rangeList, feat)
        sectionList[index] += 1
    # burst numbers
    burstNum = len(burstList)

    # percentage of incoming packets
    inPackRatio = downPackNum / (upPackNum + downPackNum)

    # number of packages
    packNum = len(tupleList)

    rtnFeat = [upStreamTotal, downStreamTotal, inPackRatio, packNum, burstNum]
    rtnFeat.extend(sectionList)

    return rtnFeat

def train(trainData, trainLabel, context):
    print('start training...')
    clf = SVC(C=context['cVal'], kernel=context['kernel'], gamma='auto', decision_function_shape='ovo', random_state=7)
    tModel = clf.fit(trainData, trainLabel)
    print('finish training...')
    return tModel

class Choices():
    def __init__(self, dataDir, model, interval):
        self.dataDir = dataDir
        self.model = model
        self.interval = interval

def generateContext():
    #acc 0.33123
    return {'cVal':127, 'kernel':'rbf'}

def main(opts):
    cVal = [115, 117, 119, 121, 123, 125, 127, 129, 131]
    kernelVal = ['rbf']

    choices = Choices(opts.input, 'Svm', 10000)
    print('extracting the features...')
    allData, allLabel, labelMap = nFoldCrossValidation.loadData(choices)
    allData = np.array(allData)
    allData = preprocessing.scale(allData)
    X_train, X_test, y_train, y_test = train_test_split(allData, allLabel, test_size=0.2, random_state=41)

    best_score = 0
    best_params = []
    for C in cVal:
        for kernel in kernelVal:
            params = [C, kernel]

            clf = SVC(C=C, kernel=kernel, gamma='auto', verbose=opts.verbose, decision_function_shape='ovo', random_state=7, max_iter=500)
            print('start training with params: {}...'.format(params))
            clf.fit(X_train, y_train)
            print('start testing...')
            predictions = clf.predict(X_test)
            acc = nFoldCrossValidation.computeACC(predictions, y_test)
            print("performance %f using %s" % (acc, params))
            if acc > best_score:
                best_score = acc
                best_params = params

    print('best: %f using %s' % (best_score, best_params))


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', help='path to training data dir')
    parser.add_argument('-m', '--modelSaveDir', help='path to model save dir')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose or not')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
