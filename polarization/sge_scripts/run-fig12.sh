#! /bin/bash
#

N_ITERATIONS=4000
for K in 1 2 3 5 10; do
    NAME="reproduce12b-$K"
    qsub -N $NAME -o "log/$NAME.log" -v K=$K -v N_ITERATIONS=$N_ITERATIONS -v OUTPUT_DIR="data/figure12b_test_3-28" run-fig12.sub
done
