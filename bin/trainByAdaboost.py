#!/usr/bin/python

import os
import sys
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
import argparse
import numpy as np

import fileUtils
import tools
import trainByVNGpp


def saveModel(modelData, fpath):
    joblib.dump(modelData, fpath)

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

def train(trainData, trainLabel):
    clf = AdaBoostClassifier(n_estimators=1200, algorithm='SAMME',
                             base_estimator=DecisionTreeClassifier(max_depth=12),
                             learning_rate=0.05, random_state=0)
    clf.fit(trainData, trainLabel)
    return clf

def main(opts):
    trainDataDir = opts.trainDataDir
    data, label = loadTrainData(trainDataDir)
    mymodel = train(data, label, opts.kmode)
    saveModel(mymodel, opts.modelSaveDir)
    print('model saved at {}'.format(opts.modelSaveDir))

def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trainDataDir', help='path to training data dir')
    parser.add_argument('-m', '--modelSaveDir', help='path to model save dir')
    parser.add_argument('-k', '--kmode', help='value for kernel mode, choose from 0-3')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
