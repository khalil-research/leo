#!/bin/bash

for i in `seq 3 7` ; do    
    make clean
    make -j NUM_OBJS=$i
    cp multiobj networkO$i
    rm multiobj
done
