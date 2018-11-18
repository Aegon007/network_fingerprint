#!/usr/bin/python

import os
import sys
import unittest
import unittest.mock
import filecmp
import shutil

sys.path.append('bin')
import trainByJaccard
import utils

WORKDATADIR = os.environ.get('WORKDATADIR')

class Test_TrainByJaccard(unittest.TestCase):
    def setUp(self):
        tmpname = os.path.basename(__file__)
        self.testDir = os.path.dirname(__file__)
        tmpname = tmpname.split('.')[0]
        tmppath = os.path.join(WORKDATADIR, tmpname)
        self.testDataDir = os.path.expandvars(tmppath)
        if not os.path.isdir(self.testDataDir):
            os.makedirs(self.testDataDir)

    def tearDown(self):
        if os.path.isdir(self.testDataDir):
            shutil.rmtree(self.testDataDir)

    def test_generateDictOfClass(self):
        test_listdir = ['/A/B/test_classA_1', '/A/B/test_classA_2', '/A/B/test_classB_1']
        result = trainByJaccard.generateDictOfClass(test_listdir)
        expect = {'test_classA':['/A/B/test_classA_1', '/A/B/test_classA_2'], 'test_classB':['/A/B/test_classB_1']}
        self.assertEquals(result, expect)

    def test_readfile(self):
        #import pdb
        #pdb.set_trace()
        pass

    @unittest.mock.patch('trainByJaccard.getListOfSet')
    def test_trainFromList(self, mock_getListOfSet):
        tmp1 = set([1, 2, 3, 4])
        tmp2 = set([2, 3, 4, 5])
        tmp3 = set([4, 5, 6, 7])
        mock_getListOfSet.return_value = [tmp1, tmp2, tmp3]
        result = trainByJaccard.trainFromList('test_fileList')
        expect = set([2, 3, 4, 5])
        self.assertEquals(result, expect)

        tmp1 = set([4, 5, 6, 7])
        mock_getListOfSet.return_value = [tmp1]
        result = trainByJaccard.trainFromList('test_fileList')
        expect = set([4, 5, 6, 7])
        self.assertEquals(result, expect)

    @unittest.mock.patch('trainByJaccard.generateDictOfClass')
    def test_train(self, mock_generateDictOfClass):
        mock_generateDictOfClass.return_value = []

    def test_writeDict2File(self):
        tmpDir = utils.makeTempDir(dir=self.testDataDir)
        tmpDict = {'key1':set([1,2,3,4]), 'key2':set([-1,-2,-3,-4])}
        trainByJaccard.writeDict2File(tmpDir, tmpDict)

        rtn = filecmp.cmpfiles(tmpDir, os.path.join(self.testDir, 'data'), ['key1', 'key2'])
        self.assertTrue(rtn)


if __name__ == "__main__":
    unittest.main()
