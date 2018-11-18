#!/usr/bin/python

import os
import sys
import unittest

sys.path.append('bin')
import testByJaccard


class Test_InferLabelByJaccardDist(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_computeJaccardDist(self):
        set1 = set([1, 2, 3])
        set2 = set([2, 3, 4])

        rtn = testByJaccard.computeJaccardDist(set1, set2)
        expect = 2/4
        self.assertEqual(rtn, expect)

    def test_getLabel(self):
        input = '/test_dir/how_many_days_untill_christmas_5_30s_1.csv'
        result = testByJaccard.getLabel(input)
        expect = 'how_many_days_untill_christmas'
        self.assertEqual(result, expect)

        input = '/test_dir/christmas_5_30s_1.csv'
        result = testByJaccard.getLabel(input)
        expect = 'christmas'
        self.assertEqual(result, expect)

        input = "/test_dir/what's_christmas_event_5_30s_1.csv"
        result = testByJaccard.getLabel(input)
        expect = "what's_christmas_event"
        #import pdb
        #pdb.set_trace()
        self.assertEqual(result, expect)

    def test_computeLabel(self):
        testFile = ''
        classFiles = []
        result = testByJaccard.computeLabel(testFile, classFiles)
        expect = ''
        self.assertEqual(result, expect)

    def test_normal_run(self):
        pass

if __name__ == "__main__":
    unittest.main()
