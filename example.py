#!/usr/bin/python

import os
import sys
import argparse


def main(opts):
    pass


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument()
    opts = parser.parse_args()
    return opts


if __name__ == "__main__":
    opts = parseOpts(sys.argv)
    main(opts)
