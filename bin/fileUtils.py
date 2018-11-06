#!/usr/bin/python

import os
import sys


def readTxtFile(fpath):
    with open(fpath, 'r') as f:
        content = f.read()
    return content


def writeTxtFile(fpath, content):
    with open(fpath, 'w') as f:
        f.write(content)


def readBinFile():
    pass


def writeBinFile():
    pass
