#!/usr/bin/python

import os
import sys
import argparse
from sklearn.externals import joblib
import numpy as np


def readfile(fpath):
    pass


def test(mymodel, testData):
    label = mymodel.predict(testData)
    return label

def main(opts):
    modelFile = ops.modelFile
    testDataFile = opts.input
    model = joblib.load(modelFile)
    testData = readfile(testDataFile)
    predicts = test(model, testData)
    print("predicted label is:{}".format(predicts))


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument()
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
