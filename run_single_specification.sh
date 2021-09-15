#!/usr/bin/env bash
# A simple bash script to run PermCheck for a single example

python -u run_perm_check.py "$@" > >(tee >(tee print.log > "logs/log_$(date).log")) 2>&1
