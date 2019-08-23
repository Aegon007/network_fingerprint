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
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import numpy as np
import pandas as pd

import fileUtils
import tools
import trainByJaccard
from trainByBayes import saveModel
import trainByVNGpp
import nFoldCrossValidation
import testByCumul


Length = 10

def padZero(theList, featLength):
    count = featLength - len(theList)
    while count > 0:
        theList.append(0)
        count = count - 1

    assert(len(theList)==featLength)
    return theList


def samplingList(cumulList, featLength):
    cLen = len(cumulList)
    if cLen <= featLength:
        newList = padZero(cumulList, featLength)
    else:
        i = 0
        interVal = int(cLen / featLength)
        newList = []
        for k in range(featLength):
            newList.append(cumulList[i])
            i = i + interVal

    #print(len(newList))
    #print(newList)
    #print(featLength)
    assert(len(newList)==featLength)
    return newList


def computeFeature(fpath, featLength):
    upPackNum, downPackNum, upStreamTotal, downStreamTotal, traceTimeList, tupleList = trainByVNGpp.readfile(fpath)

    cumulList = []
    df = pd.read_csv(fpath, sep=',', skiprows=0)
    for index, row in df.iterrows():
        pSize = row['size']
        pDirec = row['direction']
        tmp = pSize * pDirec
        if [] == cumulList:
            cumulList.append(tmp)
        else:
            lastOne = cumulList[-1] + tmp
            cumulList.append(lastOne)

    finalFeature = samplingList(cumulList, featLength)
    finalFeature.extend([upPackNum, downPackNum, upStreamTotal, downStreamTotal])

    return finalFeature


def train(trainData, trainLabel, context):
    print('start training...')
    clf = SVC(C=context['cVal'], kernel=context['kernel'], gamma='auto', decision_function_shape='ovo', random_state=7)
    #newData = preprocessing.scale(trainData)
    tModel = clf.fit(trainData, trainLabel)
    print('finish training...')
    return tModel


def generateContext():
    #[100, 'rbf', 10]
    return {'cVal':100, 'kernel':'rbf'}


class Choices():
    def __init__(self, dataDir, model, length):
        self.dataDir = dataDir
        self.model = model
        self.length = length


def main(opts):
    cVal = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    kernelVal = ['rbf', 'sigmoid', 'poly']
    len_list = [10, 30, 50, 70, 90, 110, 130]
    best_score = 0
    best_params = []
    for C in cVal:
        for kernel in kernelVal:
            for length in len_list:
                choices = Choices(opts.inputDir, 'Cumul', length)
                print("extracting the features...")
                allData, allLabel, labelMap = nFoldCrossValidation.loadData(choices)
                allData = np.array(allData)
                newAllData = preprocessing.scale(allData)
                X_train, X_test, Y_train, Y_test = train_test_split(newAllData, allLabel, test_size=0.2, random_state=7)
                clf = SVC(C=C, kernel=kernel, gamma='auto', verbose=opts.verbose, decision_function_shape='ovo', random_state=7)
                print("start training...")
                clf.fit(X_train, Y_train)
                print("start testing...")
                predictions = clf.predict(X_test)
                acc = nFoldCrossValidation.computeACC(predictions, Y_test)
                params = [C, kernel, length]
                print("performace: %f using %s" % (acc, params))
                if acc > best_score:
                    best_score = acc
                    best_params = params

    print("best: %f using %s" % (best_score, best_params))


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputDir', help='')
    parser.add_argument('-m', '--modelSaveDir', help='')
    parser.add_argument('-v', '--verbose', action='store_true', help='')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
