#!/usr/bin/python

## Copyright@Chenggang Wang
## Email: 1277223029@qq.com
## May 9th, 2019

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
import trainByVNGpp


def padZero(theList, featLength):
    count = featLength - len(theList)
    while count > 0:
        theList.append(0)
        count = count - 1

    #import pdb
    #pdb.set_trace()
    assert(len(theList)==featLength)
    return theList


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
            #print(i)
            if i > cLen:
                break
            newList.append(cumulList[i])

    assert(len(newList)==featLength)
    return newList


def computeFeature(fpath, featLength):
    upPackNum, downPackNum, upStreamTotal, downStreamTotal, traceTimeList, tupleList = trainByVNGpp.readfile(fpath)

    def readfile(line):
        tmp = line.strip().split(',')
        return tmp[-2], tmp[-1]
    cumulList = []
    for line in fileUtils.readTxtFile(fpath, 'time'):
        pSize, pDirec = readfile(line)
        tmp = fileUtils.str2int(pSize) * fileUtils.str2int(pDirec)
        if cumulList == []:
            cumulList.append(tmp)
        else:
            lastOne = cumulList[-1] + tmp
            cumulList.append(lastOne)

    finalFeature = samplingList(cumulList, featLength)
    finalFeature = finalFeature.extend([upPackNum, downPackNum, upStreamTotal, downStreamTotal])

    return finalFeature


def train(trainData, trainLabel, context):
    print('start training...')
    clf = SVC(C=context['cVal'], kernel=context['kernel'], degree=context['degree'], verbose=context['verbose'], decision_function_shape='ovo', random_state=7)
    tModel = clf.fit(trainData, trainLabel)
    print('finish training...')
    return tModel


def generateContext():
    return {'cVal':0.1, 'kernel':'rbf', 'degree':3, 'verbose':False}


def main(opts):
    trainDataDir = opts.inputDir
    data, label = loadTrainData(trainDataDir)
    import pdb
    pdb.set_trace()
    if opts.contextDict:
        context = opts.contextDict
    else:
        context = generateContext()
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
