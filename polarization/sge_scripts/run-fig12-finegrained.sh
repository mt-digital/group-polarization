#! /bin/bash
#

for K in 1 2 3 4 5 6 7 8 9 10 11 12; do
    NAME="reproduce12b-$K"
    qsub -N $NAME -o "log/$NAME.log" -v K=$K -v OUTPUT_DIR="data/finegrained_K_fm" run-fig12.sub
done
