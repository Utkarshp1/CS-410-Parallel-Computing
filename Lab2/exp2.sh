#!/bin/bash

threads=(4 9 16 25 36)
sizes=(720 1440 2160)

for p in "${threads[@]}"
do
    for n in "${sizes[@]}"
    do  
        echo $p $n "pthreads"
        ./$1 $n $p
        echo $p $n "openmp"
        ./$2 $n $p
    done
done
