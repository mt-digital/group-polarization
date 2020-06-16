# 

for S in `seq 0.5 0.05 1.0`; do
    for K in 2 3 4 5 6; do
        NAME="ic_experiment_K=$K-S=$S"
        qsub -S /bin/bash -q fast.q -cwd -j y -V -l mem_free=96G -pe smp 24 \
            -N $NAME -o "log/$NAME.log" -e "log/err/$NAME.err" \
            ~/.conda/envs/complexity/bin/polexp complexity_experiment $S $K 0.0 data/ic_experiment --n_iterations=10000 --n_trials=50
    done
done
