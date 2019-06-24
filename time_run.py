#!/usr/bin/python

import os
import subprocess
import time

dataDir = '/home/carl/work_dir/echo_proj_phase_2/data/gamma_100/gamma'
testing_methods = ['Bayes', 'VNGpp', 'Cumul', 'Adaboost']
itv_list = [1000, 10000, 0, 10000]
tmpList = []
for method, itv_val in zip(testing_methods, itv_list):
    cmd = 'python /home/carl/work_dir/network_fingerprint_proj/src/bin/nFoldCrossValidation.py -m {} -d {} -n 5 -itv {}'.format(method, dataDir, itv_val)
    print('test with cmd {}'.format(cmd))
    start = time.time()
    subprocess.run(cmd, shell=True)
    end = time.time()
    tmpLine = 'test with cmd {} using time {}'.format(cmd, str(end - start))
    tmpList.append(tmpLine)

with open('time_test.res', 'w') as f:
    content = '\n'.join(tmpList)
    content = content + '\n'
    f.write(content)
