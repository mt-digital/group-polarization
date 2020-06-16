# 
# K=2

for S in `seq 0.5 0.05 1.0`; do
    for NOISE_LEVEL in `seq 0.0 0.02 0.2`; do
        for K in 3 4 5 6 7; do
            NAME="noise_experiment_NOISE_LEVEL=$NOISE_LEVEL-S=$S-K=$K"
            qsub -S /bin/bash -q fast.q -cwd -j y -V -l mem_free=96G -pe smp 24 \
                -N $NAME -o "log/$NAME.log" -e "log/err/$NAME.err" \
                ~/.conda/envs/complexity/bin/polexp complexity_experiment \
                $S $K $NOISE_LEVEL "data/noise_experiment_K-gte-3" \
                --n_iterations=10000 --n_trials=50
        done
    done
done
