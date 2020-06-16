Also ran

```
qsub -S /bin/bash -q fast.q -cwd -j y -V -l mem_free=96G -pe smp 20 -N rerun_min_K=2 -o log/rerun_min_K=2.log -e log/err/rerun_min_K=2.err ~/.conda/envs/complexity/bin/polexp rerun_experiment "/scratch/mturner8/finegrained_K_4-18/" "K=2" "/scratch/mturner8/final_rerun_minpol_K=2.hdf" --trial_index=min --n_trials=100 --n_iterations=10000
```

to do the re-run starting from the trial that ended with the minimal 
