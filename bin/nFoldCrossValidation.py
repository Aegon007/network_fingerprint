#!/usr/bin/python

import os
import sys
import argparse
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
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

import utils
import fileUtils
import parseWord2VecFile


def computeACC(predictions, labels):
    assert(len(predictions)==len(labels))
    count = 0
    l = len(predictions)
    for i in range(l):
        if predictions[i] == labels[i]:
            count = count + 1
    return count/l


def getFeature(fpath, opts):
    if opts.model == 'Jaccard':
        return fpath
    elif opts.model == 'Bayes':
        #rangeList = [-1500, 1501, 50]
        rangeList = [-1500, 1501, int(opts.interval)]
        return trainByBayes.computeFeature(fpath, rangeList)
    elif opts.model == 'VNGpp':
        #rangeList = [-200000, 200001, 5000]
        rangeList = [-4500000, 4500001, int(opts.interval)]
        return trainByVNGpp.computeFeature(fpath, rangeList)
    elif opts.model == 'Svm':
        #rangeList = [-200000, 200001, int(opts.interval)]
        rangeList = [-4500000, 4500001, int(opts.interval)]
        return trainBySvm.computeFeature(fpath, rangeList)
    elif opts.model == 'Adaboost':
        #rangeList = [-200000, 200001, int(opts.interval)]
        rangeList = [-4500000, 4500001, int(opts.interval)]
        return trainByAdaboost.computeFeature(fpath, rangeList)
    elif opts.model == 'Cumul':
        return trainByCumul.computeFeature(fpath, trainByCumul.Length)
    else:
        raise ValueError('input is should among Jaccard/Bayes/VNGpp/Svm/Adaboost')


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

def fileIsEmpty(fpath):
    with open(fpath, 'r') as f:
        content = f.read()
    return True if content=='' else False

def loadData(dataDir, opts):
    subDirs = fileUtils.getSubDirs(dataDir)
    tmpDataList = []
    tmpLabelList = []
    fList = []
    dirRoot = os.path.abspath(dataDir)
    for subDir in subDirs:
        dirpath = os.path.join(dirRoot, subDir)
        tmpList = fileUtils.genfilelist(dirpath)
        fList.extend(tmpList)

    """
    for item in fList.copy():
        tmp = os.path.basename(item)
        cname = testByJaccard.getLabel(tmp)
        if cname == 'what_are_the_most_popular_books_this_week':
            fList.remove(item)
    """

    labelMap = getLabelMap(fList)
    totalNum = len(fList)
    count = 1
    for fp in fList:
        if fileIsEmpty(fp):
            print('skip empty file {}'.format(fp))
            continue
        tmpData = getFeature(fp, opts)
        tmpDataList.append(tmpData)
        tmpLabel = mapLabel(fp, labelMap)
        tmpLabelList.append(tmpLabel)
        print('\r now reading {:d}/{:d}'.format(count, totalNum), end='')
        count = count + 1

    allData = np.array(tmpDataList)
    allLabel = np.array(tmpLabelList)

    return allData, allLabel, labelMap


def convert2Nums(predictions, labelMap):
    num_predicts = []
    for item in predictions:
        numLabel = labelMap[item]
        num_predicts.append(numLabel)

    return num_predicts


def convert2Str(numLabels, labelMap):
    tmpList = []
    for num in numLabels:
        for key in labelMap.keys():
            if num == labelMap[key]:
                tmpList.append(key)
                break
    return tmpList


def computeDistance(vecA, vecB, method):
    npVecA = np.array(vecA)
    npVecB = np.array(vecB)
    if 'cosin' == method:
        return np.dot(npVecA, npVecB)/(np.linalg.norm(npVecA)*np.linalg.norm(npVecB))
    elif 'euclidean' == method:
        return np.linalg.norm(npVecA - npVecB)
    else:
        raise ValueError('method {} not supported yet'.format(method))

def sortTupleList(tupleList):
    return sorted(tupleList, key=lambda x: x[1])

def computeRankScore(word2vecDict, prediction, label):
    vec_pred = word2vecDict[prediction]
    scoreList = []
    for key in word2vecDict.keys():
        if key == prediction:
            continue
        vec_i = word2vecDict[key]
        score = computeDistance(vec_i, vec_pred, 'cosin')
        tmpTuple = (key, score)
        scoreList.append(tmpTuple)

    sortedList = sortTupleList(scoreList)
    for i in range(len(sortedList)):
        if label == sortedList[i][0]:
            return i
    return 0

def writeTestResults(str_Y_test, str_predictions, opts, acc_list, avg_accuracy):
    word2vecfile = opts.word2vec
    file2store = opts.file2store
    word2vecDict = parseWord2VecFile.loadData(word2vecfile)
    assert(len(str_Y_test) == len(str_predictions))
    tmpList = ['label\tprediction\tdistance\trank score']

    checkList1 = []
    checkList2 = []
    for item in str_Y_test:
        if item in word2vecDict.keys():
            continue
        checkList1.append(item)

    for item in word2vecDict.keys():
        if item in str_Y_test:
            continue
        checkList2.append(item)

    if len(checkList1)>0 or len(checkList2)>0:
        print('word in label but not in dict')
        print(checkList1)
        print('=====================')
        print('word in dict but not in label')
        print(checkList2)
        raise ValueError('checkList is not empty')

    scoreList = []
    rankscoreList = []
    for i in range(len(str_Y_test)):
        item_Y = str_Y_test[i]
        item_pred = str_predictions[i]
        vec_Y = word2vecDict[item_Y]
        vec_pred = word2vecDict[item_pred]
        score = computeDistance(vec_Y, vec_pred, 'cosin')
        scoreList.append(score)
        rankscore = computeRankScore(word2vecDict, item_pred, item_Y)
        rankscoreList.append(rankscore)
        tmpLine = '{}\t{}\t{}\t{}'.format(item_Y, item_pred, score, rankscore)
        tmpList.append(tmpLine)

    tmpLine = '\n\n=========== Statistics =========\n'
    tmpList.append(tmpLine)
    sumUp = sum(scoreList)
    average = sumUp/len(scoreList)
    tmpLine = 'total = {};\taverage = {}'.format(sumUp, average)
    tmpList.append(tmpLine)

    aveRankScore = np.mean(rankscoreList)
    tmpLine = 'average rank score is: {}'.format(str(aveRankScore))
    print(tmpLine)
    tmpList.append(tmpLine)

    varRankScore = np.std(rankscoreList)
    tmpLine = 'variance of rank score is: {}'.format(str(varRankScore))
    print(tmpLine)
    tmpList.append(tmpLine)

    tmpLine = 'acc result for each test round is: {}'.format(str(acc_list))
    tmpList.append(tmpLine)
    tmpLine = 'prediction with method {}, has a accuracy is: {}'.format(opts.model, str(avg_accuracy))
    tmpList.append(tmpLine)
    tmpLine = 'accuracy variance is: {}'.format(np.std(acc_list))
    tmpList.append(tmpLine)

    content = '\n'.join(tmpList)
    print(content)
    with open(file2store, 'w') as f:
        f.write(content)

    return aveRankScore


def main(opts):
    print("start extracting feature...")
    allData, allLabel, labelMap = loadData(opts.dataDir, opts)
    newData = preprocessing.scale(allData)
    skf = StratifiedKFold(n_splits=int(opts.nFold))
    acc_list = []
    train_time_list = []
    test_time_list = []
    print("selecting model {}".format(opts.model))
    for train_index, test_index in skf.split(newData, allLabel):
        print("start a round of testing...")
        X_train, X_test = newData[train_index], newData[test_index]
        Y_train, Y_test = allLabel[train_index], allLabel[test_index]

        if opts.model == 'Jaccard':
            modelFileDir = utils.makeTempDir()
            trainStart = time.time()
            trainByJaccard.train(X_train, modelFileDir)
            trainEnd = time.time()
            testStart = time.time()
            str_predictions = testByJaccard.test(X_test, modelFileDir)
            testEnd = time.time()
            predictions = convert2Nums(str_predictions, labelMap)
        elif opts.model == 'Bayes':
            trainStart = time.time()
            model = trainByBayes.train(X_train, Y_train)
            trainEnd = time.time()
            testStart = time.time()
            predictions = testByBayes.test(model, X_test)
            testEnd = time.time()
            str_predictions = convert2Str(predictions, labelMap)
        elif opts.model == 'VNGpp':
            trainStart = time.time()
            model = trainByVNGpp.train(X_train, Y_train)
            trainEnd = time.time()
            testStart = time.time()
            predictions = testByVNGpp.test(model, X_test)
            testEnd = time.time()
            str_predictions = convert2Str(predictions, labelMap)
        elif opts.model == 'Svm':
            context = trainBySvm.generateContext()
            trainStart = time.time()
            model = trainBySvm.train(X_train, Y_train, context)
            trainEnd = time.time()
            testStart = time.time()
            predictions = testBySvm.test(model, X_test)
            testEnd = time.time()
            str_predictions = convert2Str(predictions, labelMap)
        elif opts.model == 'Adaboost':
            trainStart = time.time()
            model = trainByAdaboost.train(X_train, Y_train)
            trainEnd = time.time()
            testStart = time.time()
            predictions = testByAdaboost.test(model, X_test)
            testEnd = time.time()
            str_predictions = convert2Str(predictions, labelMap)
        elif opts.model == 'Cumul':
            context = trainByCumul.generateContext()
            trainStart = time.time()
            model = trainByCumul.train(X_train, Y_train, context)
            trainEnd = time.time()
            testStart = time.time()
            predictions = testByCumul.test(model, X_test)
            testEnd = time.time()
            str_predictions = convert2Str(predictions, labelMap)
        else:
            raise ValueError('input is should among Jaccard/Bayes/VNGpp')

        accuracy = computeACC(predictions, Y_test)
        train_time = trainEnd - trainStart
        test_time = testEnd - testStart

        acc_list.append(accuracy)
        train_time_list.append(train_time)
        test_time_list.append(test_time)

    print(acc_list)
    avg_accuracy = sum(acc_list)/len(acc_list)
    avg_train_time = sum(train_time_list)/len(train_time_list)
    print('prediction with method {}, has a accuracy is: {}, with average training time {}'.format(opts.model, avg_accuracy, str(avg_train_time)))

    if opts.word2vec:
        str_Y_test = convert2Str(Y_test, labelMap)
        aveRankScore = writeTestResults(str_Y_test,
                                        str_predictions,
                                        opts, acc_list, avg_accuracy)
        return aveRankScore, avg_accuracy


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',
                        help='choose model from Jaccard/Bayes/VNGpp/Svm/Adaboost/Cumul')
    parser.add_argument('-d', '--dataDir', help='data dir where store all data')
    parser.add_argument('-n', '--nFold', help='indicate how many fold')
    parser.add_argument('-w', '--word2vec', help='file store word2vec data')
    parser.add_argument('-f', '--file2store', help='file store result data')
    parser.add_argument('-itv', '--interval', help='set interval value')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
