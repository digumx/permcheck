#!/usr/bin/env bash
# Runs all the experiments

cd test-networks
python test_acc.py 10000 ../accuracy.log
cd ..
python -u run_perm_check_experiments.py perm_check_results.log "$@" > >(tee >(tee print.log > "logs/perm_check_log_$(date).log")) 2>&1
python -u run_marabou_experiments.py marabou_results.log "$@" > >(tee >(tee print.log > "logs/marabou_log_$(date).log")) 2>&1
