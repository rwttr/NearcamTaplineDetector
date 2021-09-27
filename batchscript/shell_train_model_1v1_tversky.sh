#!/bin/bash
julia2='/home/rattachai/julia/julia-1.6.3/bin/julia --threads 8'
$julia2 train_model_1_std_std_tversky_cmd.jl --fold_no=1
$julia2 train_model_1_std_std_tversky_cmd.jl --fold_no=2 
$julia2 train_model_1_std_std_tversky_cmd.jl --fold_no=3 
$julia2 train_model_1_std_std_tversky_cmd.jl --fold_no=4  
$julia2 train_model_1_std_std_tversky_cmd.jl --fold_no=5
