#! /bin/bash

export PATH=$PATH:bin

cat test/test.list | while read line
do
    echo "========== now testing file: ${line} ============="
    python3.6 ${line}
    echo "============================================================="
    echo
    echo
done
