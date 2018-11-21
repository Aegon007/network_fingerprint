#!/usr/bin/python

import os
import sys
import unittest
import unittest.mock

sys.path.append('bin')
import trainByVNGpp


class Test_nFoldCrossValidatdation(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_calculateBursts(self):
        testTupleList = [(10, 1),
                         (20, -1),
                         (30, -1),
                         (15, 1),
                         (20, 1),
                         (60, 1),
                         (70, -1),
                         (11, -1)]
        result = trainByVNGpp.calculateBursts(testTupleList)
        expect = [10, -50, 95, -81]
        self.assertEquals(result, expect)

    @unittest.mock.patch('trainByVNGpp.readfile')
    def test_computeFeature(self, mock_readfile):
        #import pdb
        #pdb.set_trace()
        mock_readfile.return_value = 10, 10, [13.1, 15.0, 11.004, 14.99, 16.11, 11.001], [(51, -1), (99, -1), (100, 1), (101, -1), (150, 1), (79, -1), (0, -1), (50, 1)]
        rangeList = [-150, 151, 50]
        result = trainByVNGpp.computeFeature('test_file', rangeList)
        expect = [5.109, 10, 10, 2, 1, 0, 0, 1, 2]
        self.assertEquals(result, expect)


if __name__ == "__main__":
    unittest.main()
