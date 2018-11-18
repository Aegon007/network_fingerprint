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


def calculateBursts():
    pass


def computeFeature(fpath):
    UpStreamTotal = 0
    DownStreamTotal = 0
    UpStreamPackageNum = 0
    DownStreamPackageNum = 0
    traceTimeList = []
    for line in fileUtils.readTxtFile(fpath, ',time'):
        tmp = line.split(',')
        flag = fileUtils.str2int(tmp[-1])
        traceTimeList.append(tmp[1])
        if 1 == flag:
            UpStreamPackageNum += 1
            UpStreamTotal += fileUtils.str2int(tmp[-2])
        elif -1 == flag:
            DownStreamPackageNum += 1
            DownStreamTotal += fileUtils.str2int(tmp[-2])
        else:
            raise ValueError('unexpected flag value: {}'.format(flag))

    traceTimeList.sort()
    TotalTraceTime = traceTimeList[-1] - traceTimeList[0]
    TotalBurstByte = DownStreamTotal + UpStreamTotal

    return [TotalTraceTime, TotalBurstByte, UpStreamPackageNum, UpStreamTotal, DownStreamPackageNum, DownStreamTotal]


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
