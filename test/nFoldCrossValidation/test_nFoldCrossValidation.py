#!/usr/bin/python

import os
import sys
import unittest

sys.path.append('bin')
import nFoldCrossValidation


class Test_nFoldCrossValidatdation(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_computeACC(self):
        a = [1,2,3,4,5,1,2,3,4,5]
        b = [1,2,3,3,5,1,2,3,4,4]
        result = nFoldCrossValidation.computeACC(a, b)
        expect = 0.8
        self.assertEqual(result, expect)

    def test_getLabelMap(self):
        tmpList = ['a_aaa_1.py', 'a_aaa_2.py', 'b_bbb_2.py', 'c_ccc_1.py', 'c_ccc_2.py']
        result = nFoldCrossValidation.getLabelMap(tmpList)
        expect = {'a_aaa':1, 'b_bbb':2, 'c_ccc':3}
        self.assertEqual(result, expect)

    def test_computeDistance(self):
        vecA = [1, 1, 1]
        vecB = [1, 1, 1]
        result = nFoldCrossValidation.computeDistance(vecA, vecB, 'cosin')
        expect = 1
        self.assertAlmostEqual(result, expect)

        vecA = [1, 2]
        vecB = [2, 1]
        result = nFoldCrossValidation.computeDistance(vecA, vecB, 'cosin')
        expect = 0.8
        self.assertAlmostEqual(result, expect)

    def test_convert2Str(self):
        diction = {'aaa':1, 'bbb':2, 'ccc':3, 'ddd':4}
        numList = [3, 2, 1, 4]
        #import pdb
        #pdb.set_trace()
        result = nFoldCrossValidation.convert2Str(numList, diction)
        expect = ['ccc', 'bbb', 'aaa', 'ddd']
        self.assertEqual(result, expect)

    def test_sortTupleList(self):
        tupleList = [('aaa', 6), ('ccc', 1), ('bbb', 10)]
        result = nFoldCrossValidation.sortTupleList(tupleList)
        expect = [('ccc', 1), ('aaa', 6), ('bbb', 10)]
        self.assertEqual(result, expect)

    def test_computeRankScore(self):
        word2vecDict = {'aaa':[0.6, 0.7, 0.8], 'bbb':[1.2, 0.7, 1.1], 'ccc':[0.9, 1.0, 3.2], 'ddd':[0.5, 0.4, 0.9]}
        prediction = 'aaa'
        label = 'bbb'
        result = nFoldCrossValidation.computeRankScore(word2vecDict, prediction, label)
        expect = 2
        self.assertEqual(result, expect)


if __name__ == "__main__":
    unittest.main()
