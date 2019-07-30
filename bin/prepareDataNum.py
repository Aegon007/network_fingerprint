#encoding=utf-8
#!/usr/bin/python

import os
import sys
import numpy as np
import pandas as pd
import re

from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import fileUtils
import nFoldCrossValidation
import testByJaccard


def loadData(opts, num4class):
    subDirs = os.listdir(opts.input)
    fList = []
    for subDir in subDirs:
        dpath = os.path.join(opts.input, subDir)
        tmpList = fileUtils.genfilelist(dpath)
        fList.extend(tmpList)

    labelMap = nFoldCrossValidation.getLabelMap(fList)
    classCountDict = defaultdict(int)
    for key in labelMap.keys():
        classCountDict[key] = 0

    tmpDataList = []
    tmpLabelList = []
    for fp in fList:
        if 0 == os.path.getsize(fp):
            print('skip empty file {}'.format(fp))
            continue
        label = testByJaccard.getLabel(fp)
        if classCountDict[label] >= num4class:
            continue
        else:
            classCountDict[label] = classCountDict[label] + 1
        tmpData = nFoldCrossValidation.getFeature(fp, opts)
        tmpDataList.append(tmpData)
        tmpLabel = nFoldCrossValidation.mapLabel(fp, labelMap)
        tmpLabelList.append(tmpLabel)

    allData = np.array(tmpDataList)
    allLabel = np.array(tmpLabelList, dtype=np.uint8)

    return allData, allLabel, labelMap

def verifyData(allLabel, num4class):
    from collections import Counter
    tmpDict = Counter(allLabel)
    flag = True
    for key in tmpDict.keys():
        if tmpDict[key] != num4class:
            print('key {} of data is not get correct number'.format(key))
            flag = False

    if flag:
        print('all data extracted are correct, con!')

    return flag

def loadAndVerifyData(droot, data_dim, num4class, dataType):
    allData, allLabel, labelMap = loadData(droot, data_dim, num4class, dataType)
    if not verifyData(allLabel, num4class):
        raise
    return allData, allLabel, labelMap
