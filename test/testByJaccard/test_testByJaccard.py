#!/usr/bin/python

import os
import sys
import unittest

sys.path.append('bin')
import inferLabelByJaccardDist


class Test_InferLabelByJaccardDist(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_computeJaccardDist(self):
        set1 = set([1, 2, 3])
        set2 = set([2, 3, 4])

        rtn = inferLabelByJaccardDist.computeJaccardDist(set1, set2)
        expect = 2/4
        self.assertEqual(rtn, expect)

    def test_getLabel(self):
        input = '/test_dir/how_many_days_untill_christmas_5_30s_1.csv'
        #import pdb
        #pdb.set_trace()
        result = inferLabelByJaccardDist.getLabel(input)
        expect = 'how_many_days_untill_christmas'
        self.assertEqual(result, expect)

    def test_computeLabel(self):
        testFile = ''
        classFiles = []
        result = inferLabelByJaccardDist.computeLabel(testFile, classFiles)
        expect = ''
        self.assertEqual(result, expect)

    def test_normal_run(self):
        pass

if __name__ == "__main__":
    unittest.main()
