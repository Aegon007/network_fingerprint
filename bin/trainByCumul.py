#!/usr/bin/python

import os
import sys
import argparse
import math
from collections import defaultdict
from sklearn.svm import SVC
import numpy as np

import fileUtils
import tools
import trainByJaccard
from trainByBayes import saveModel


def samplingList(cumulList, featLength):
    cLen = len(cumulList)
    if cLen <= featLength:
        newList = padZero(cumulList, featLength)
    else:
        i = 0
        interVal = int(cLen / featLength) + 1
        newList = [cumulList[0]]
        while 1:
            i = i + interVal
            print(i)
            if i > cLen:
                break
            newList.append(cumulList[i])

    return newList


def computeFeature(fpath, featLength):
    upPackNum, downPackNum, upStreamTotal, downStreamTotal, traceTimeList, tupleList = trainByVNGpp.readfile(fpath)

    cumulList = []
    with open(fpath, 'r') as f:
        for line in f:
            pSize, pDirec = readfile(line)
            tmp = int(pSize) * int(pDirec)
            if cumulList == []:
                cumulList.append[tmp]
            else:
                lastOne = cumulList[-1] + tmp
                cumulList.append(lastOne)

    finalFeature = samplingList(cumulList, featLength)
    finalFeature = finalFeature.extend([upPackNum, downPackNum, upStreamTotal, downStreamTotal])

    return finalFeature


def train(trainData, trainLabel, context):
    clf = SVC(c=context.cVal, kernel=context.kernel, degree=context.degreeVal, verbose=context.verbose, decision_function_shape='ovo', random_state=7)
    tModel = clf.fit(trainData, trainLabel)
    return tModel


def generateContext(opts):
    cDict = opts.contextDict
    return cDict


def main(opts):
    trainDataDir = opts.inputDir
    data, label = loadTrainData(trainDataDir)
    context = generateContext(opts)
    mymodel = train(data, label, context)
    saveModel(mymodel, opts.modelSaveDir)
    print("model saved at {}".format(opts.modelSaveDir))


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputDir', help='')
    parser.add_argument('-m', '--modelSaveDir', help='')
    parser.add_argument('-c', '--contextDict', help='')
    parser.add_argument('-v', '--verbose', help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
