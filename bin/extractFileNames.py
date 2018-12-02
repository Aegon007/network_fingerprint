#!/usr/bin/python

import os
import sys
import argparse

import fileUtils
import testByJaccard


def main(opts):
    fileList = fileUtils.genfilelist(opts.dirpath)
    count = 0
    with open(opts.respath, 'w') as f:
        for fpath in fileList:
            fname = os.path.basename(fpath)
            #fname = testByJaccard.getLabel(fname)
            tmpLine = '{}\n'.format(fname)
            f.write(tmpLine)
            count = count + 1
    print('write {} file names'.format(count))

def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dirpath', help='dir path to those files')
    parser.add_argument('-r', '--respath', help='result path to store result')
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
