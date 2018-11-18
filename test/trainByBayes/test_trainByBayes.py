#!/usr/bin/python

import os
import sys
import unittest

sys.path.append('bin')
import trainByBayes


class Test_nFoldCrossValidatdation(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_getSectionList(self):
        #import pdb
        #pdb.set_trace()
        result1, result2 = trainByBayes.getSectionList(-100, 100, 50)
        expect1 = [-100, -50, 0, 50, 100]
        expect2 = 4
        self.assertEqual(result1, expect1)
        self.assertEqual(result2, expect2)

    def test_computeRange(self):
        rangeList = [-100, -50, 0, 50, 100]
        feature = -55
        result = trainByBayes.computeRange(rangeList, feature)
        expect = 0
        self.assertEqual(result, expect)

        rangeList = [-100, -50, 0, 50, 100]
        feature = 55
        result = trainByBayes.computeRange(rangeList, feature)
        expect = 3
        self.assertEqual(result, expect)

    def test_computeFeature(self):
        pass

    def test_loadTrainData(self):
        pass

if __name__ == "__main__":
    unittest.main()
