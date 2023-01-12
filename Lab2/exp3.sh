#!/bin/bash

threads=(2 4 8 16 32)
sizes=(250 500 1000 2000 4000)
chunk=(50 60 100 200)

for p in "${threads[@]}"
do
    for n in "${sizes[@]}"
    do  
        for c in "${chunk[@]}"
        do 
            echo "Running with $p threads and $n size with chunk size of $c for static scheduling ..."
            ./$1 $n $p $c 0
            echo "Running with $p threads and $n size with chunk size of $c for dynamic scheduling ..."
            ./$1 $n $p $c 1
        done
    done
done
