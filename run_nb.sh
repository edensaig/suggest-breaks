#! /bin/bash
in_file=$1.ipynb
out_file=$1_$2.ipynb
seed=$2
RANDOM_STATE=${seed} conda run -n lvml --no-capture-output jupyter nbconvert --to notebook --execute ${in_file} --output ${out_file}
# Example: for i in {1..10}; do echo dispatching run with seed $i; ./run_nb.sh movielens_experiment $i &; done
