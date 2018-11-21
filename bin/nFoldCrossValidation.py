#!/usr/bin/python

import os
import sys
import argparse
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import numpy as np
from collections import defaultdict

import trainByBayes
import trainByJaccard
import trainByVNGpp

import testByBayes
import testByJaccard
import testByVNGpp

import utils
import fileUtils


def computeACC(predictions, labels):
    assert(len(predictions)==len(labels))
    count = 0
    l = len(predictions)
    for i in range(l):
        if predictions[i] == labels[i]:
            count = count + 1
    return count/l


def getFeature(fpath):
    global opts
    if opts.model == 'Jaccard':
        return fpath
    elif opts.model == 'Bayes':
        rangeList = [-1500, 1501, 50]
        return trainByBayes.computeFeature(fpath, rangeList)
    elif opts.model == 'VNGpp':
        rangeList = [-50000, 50001, 5000]
        return trainByVNGpp.computeFeature(fpath, rangeList)
    else:
        raise ValueError('input is should among Jaccard/Bayes/VNGpp')


def getLabelMap(fnameList):
    cNameDict = defaultdict(int)
    count = 1
    for fname in fnameList:
        cname = testByJaccard.getLabel(fname)
        if cname in cNameDict.keys():
            continue
        else:
            cNameDict[cname] = count
            count += 1

    return cNameDict


def mapLabel(fpath, labelMap):
    fname = os.path.basename(fpath)
    fname = testByJaccard.getLabel(fname)
    return labelMap[fname]


def loadData(dataDir):
    #tmpList = os.listdir(dataDir)
    #fList = list(map(lambda x: os.path.join(dataDir, x), tmpList))
    fList = fileUtils.genfilelist(dataDir)
    labelMap = getLabelMap(fList)
    tmpDataList = []
    tmpLabelList = []
    for fp in fList:
        tmpData = getFeature(fp)
        tmpDataList.append(tmpData)
        tmpLabel = mapLabel(fp, labelMap)
        tmpLabelList.append(tmpLabel)

    allData = np.array(tmpDataList)
    allLabel = np.array(tmpLabelList)

    return allData, allLabel, labelMap


def convert2Nums(predictions, labelMap):
    num_predicts = []
    for item in predictions:
        numLabel = labelMap[item]
        num_predicts.append(numLabel)

    return num_predicts


def main(opts):
    allData, allLabel, labelMap = loadData(opts.dataDir)
    skf = StratifiedKFold(n_splits=int(opts.nFold))
    acc_list = []
    for train_index, test_index in skf.split(allData, allLabel):
        X_train, X_test = allData[train_index], allData[test_index]
        Y_train, Y_test = allLabel[train_index], allLabel[test_index]

        if opts.model == 'Jaccard':
            modelFileDir = utils.makeTempDir()
            trainByJaccard.train(X_train, modelFileDir)
            #import pdb
            #pdb.set_trace()
            predictions = testByJaccard.test(X_test, modelFileDir)
            predictions = convert2Nums(predictions, labelMap)
        elif opts.model == 'Bayes':
            model = trainByBayes.train(X_train, Y_train)
            predictions = testByBayes.test(model, X_test)
        elif opts.model == 'VNGpp':
            model = trainByVNGpp.train(X_train, Y_train)
            predictions = testByVNGpp.test(model, X_test)
        else:
            raise ValueError('input is should among Jaccard/Bayes/VNGpp')

        accuracy = computeACC(predictions, Y_test)
        acc_list.append(accuracy)
    print(acc_list)
    avg_accuracy = sum(acc_list)/len(acc_list)
    print('prediction with method {}, has a accuracy is: {}'.format(opts.model, avg_accuracy))


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='choose which model Jaccard/Bayes/VNGpp you want to use')
    parser.add_argument('-d', '--dataDir', help='data dir where store all data')
    parser.add_argument('-n', '--nFold', help='indicate how many fold')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
