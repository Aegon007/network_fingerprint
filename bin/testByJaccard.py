#!/usr/bin/python

import os
import sys
import argparse
import re

import fileUtils
import csvList2Set
import trainByJaccard


def computeJaccardDist(setA, setB):
    intersec_set = setA.intersection(setB)
    union_set = setA.union(setB)
    return len(intersec_set) / len(union_set)


def getLabel(fpath):
    '''need to negotiate the file name pattern first'''
    pattern = '(.*_[a-zA-Z]+)_[0-9].*'
    fname = os.path.basename(fpath)
    m = re.match(pattern, fname)
    if m:
        return m.group(1)
    else:
        return ''


def computeLabel(testFile, classFiles):
    max_value = 0
    max_file = ''
    testData = trainByJaccard.readfile(testfile)
    for cfile in classFiles:
        classData = trainByJaccard.readfile(cfile)
        tmp_value = computeJaccardDist(testData, classData)
        if tmp_value > max_value:
            max_value = tmp_value
            max_file = cfile

    label = getLabel(max_file)

    print('the label for given test file {} is: {}'.format(testfile, label))

    return label


def main(opts):
    testFiles = genfilelist(opts.testFilePath)
    classFiles = genfilelist(opts.classFilePath)

    for testFile in testFiles:
        label = computeLabel(testFile, classFiles)

    accuracy = computeAccuracy()
    recall = computeRecall()


def parseOpts():
    parser = argparse.ArgumentParser()
    parser.add_argument()
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
