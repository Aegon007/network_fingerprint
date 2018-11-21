#!/usr/bin/python

import os
import sys
import unittest
import unittest.mock

sys.path.append('bin')
import trainByBayes


class Test_nFoldCrossValidatdation(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass


    @unittest.mock.patch('trainByBayes.readfile')
    def test_computeFeature(self, mock_readfile):
        mock_readfile.return_value = [50, 99, 98, 66, 5, -90, -80, -77, 44, -44]
        rangeList = [-100, 100, 50]
        result = trainByBayes.computeFeature('test_file', rangeList)
        expect = [3, 1, 2, 4]
        self.assertTrue(4==len(result))
        self.assertEquals(result, expect)


    def test_loadTrainData(self):
        pass

if __name__ == "__main__":
    unittest.main()
