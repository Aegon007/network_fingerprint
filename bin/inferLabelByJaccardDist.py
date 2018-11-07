#!/usr/bin/python

import os
import sys
import argparse

import fileUtils
import csvList2Set


def computeJaccardDist(setA, setB):
    intersec_set = setA.intersection(setB)
    union_set = setA.update(setB)
    return len(intersec_set) / len(union_set)


def computeLabel(testFile, classFiles):
    max_value = 0
    max_file = ''
    for cfile in classFiles:
        tmp_value = computeJaccardDist(testfile, cfile)
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
