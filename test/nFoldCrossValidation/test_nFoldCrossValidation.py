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
        self.assertEquals(result, expect)


if __name__ == "__main__":
    unittest.main()
