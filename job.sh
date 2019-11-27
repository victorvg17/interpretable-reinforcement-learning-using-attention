#!/bin/sh
#$ -cwd
#$ -l long
#$ -l gpus=1
#$ -e ./logs/
#$ -o ./logs/
mkdir -p ./logs/
mkdir -p ./runs/
mkdir -p ./output/

python -m torchbeast.monobeast \
       --xpid cpu \
       --num_actors 10 \
       --num_threads 2 \
       --total_steps 2_000_000_000 \
       --learning_rate 0.0002 \
       --grad_norm_clipping 1280 \
       --epsilon 0.01 \
       --entropy_cost 0.01 \
       --xpid "run_001" \
       --savedir "./output" \
       --unroll_length 50 --batch_size 32
