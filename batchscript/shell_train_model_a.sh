#!/bin/bash
julia2='/home/rattachai/julia/julia-1.6.3/bin/julia --threads 4'
gpu_device=0
$julia2 train_model_a_cmd.jl --fold_no=1 --gpu_id=$gpu_device
$julia2 train_model_a_cmd.jl --fold_no=2 --gpu_id=$gpu_device
$julia2 train_model_a_cmd.jl --fold_no=3 --gpu_id=$gpu_device
$julia2 train_model_a_cmd.jl --fold_no=4 --gpu_id=$gpu_device
$julia2 train_model_a_cmd.jl --fold_no=5 --gpu_id=$gpu_device