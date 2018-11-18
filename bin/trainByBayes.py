#!/usr/bin/python

import os
import sys
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
import argparse
import numpy as np

import fileUtils


def saveModel(modelData, fpath):
    joblib.dump(modelData, fpath)


def getSectionList(start, end, interval):
    rangeList = [start]
    while 1:
        tmpPoint = start + interval
        if tmpPoint >= end:
            break
        rangeList.append(tmpPoint)
        start = tmpPoint
    rangeList.append(end)
    secList = len(range(len(rangeList)-1))
    return rangeList, secList


def computeRange(rangeList, feature):
    l = len(rangeList) - 1
    for i in range(l):
        x1 = rangeList[i]
        x2 = rangeList[i+1]
        if x1 <= feature < x2:
            return i

    raise ValueError('the value of feature exceed the rangeList')


def computeFeature(fpath):
    rangeList, sectionList = getSectionList(-1500, 1500, 100)
    features = readfile(fpath)
    for feat in features:
        index = computeRange(rangeList, feat)
        sectionList[index] += 1

    return sectionList


def computeAllFeature(dpath):
    fileList = fileUtils.genfilelist(dpath)
    allFeatures = []
    for fpath in fileList:
        tmpFeat = computeOneSampleFeature(fpath)
        allFeatures.append(tmpFeat)

    return np.array(allFeatures)


def train(trainData, trainLabel):
    gnb = GaussianNB()
    y_pred = gnb.fit(trainData, trainLabel)
    return y_pred


def loadTrainData(dataDir):
    fList = fileUtils.genfilelist(dataDir)


def main(opts):
    trainDataDir = opts.trainDataDir
    data, label = loadTrainData(trainDataDir)
    mymodel = train(data, label)
    saveModel(mymodel, fpath)


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trainDataDir', help='path to training data dir')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
