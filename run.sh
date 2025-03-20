#!/bin/bash

BASE_ARGS="--network_depth 2 --network_width 64 --num_episodes 200 --num_warmup_episodes 100 --num_episode_steps 500 --num_train_epochs 50 --num_simulations 500 --max_depth 20 --gumbel_scale 1.0 --learning_rate 1e-4 --batch_size 256 --steps 10"

declare -a commands=(
    "--name baseline"
)

for args in "${commands[@]}"; do
    echo "Launching experiment with args: $args"
    python main.py $BASE_ARGS $args
    echo "Finished: $args"
    sleep 10
done
