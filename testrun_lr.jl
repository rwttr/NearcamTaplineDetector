# testbench for learning rate scheduler with resumeable options
# warmup - decay lr in separate function

# Training.jl
## load dataset
include("NearcamTaplineDataset.jl")
import .NearcamTaplineDataset as NDS

# Network core
using Flux
using Flux:Data.DataLoader
using Flux:@epochs
using CUDA

# Utils
using Statistics
using ImageView
using Logging
using TensorBoardLogger
using BSON:@save
using BSON:@load
using ParameterSchedulers
using Functors
using ProgressMeter


# save / model checkpoint enable
chkpoint_enable = true

log_enable = true
if log_enable
    # Setup Log for Training Loss
    logger = TBLogger("content/log", tb_overwrite);
    tensorboard_cmd = `tensorboard --logdir=content`
end

# Learning rate scheduler function
let # scope for state variable
    local warmUpEpoch # state var
    global scheduleLR

    function scheduleLR(iter_count, epoch, lr, warmupPeriod, nepoch)        
        
        if iter_count <= warmupPeriod
            # Increase the learning rate for number of iterations in warmup period.
            schedule_lr = lr * ((iter_count / warmupPeriod)^4);
            warmUpEpoch = epoch;
        elseif iter_count >= warmupPeriod && 
            epoch < warmUpEpoch + floor(0.5 * (nepoch - warmUpEpoch))
            # After warm up period, keep the learning rate constant
            schedule_lr = lr;        
        elseif epoch >= warmUpEpoch + floor(0.5 * (nepoch - warmUpEpoch)) && 
            epoch < warmUpEpoch + floor(0.9 * (nepoch - warmUpEpoch))
            # If the remaining number of epochs > 60 percent and < 90 percent
            schedule_lr = lr * 0.25;        
        else
            # If remaining epochs > 90 percent
            schedule_lr = lr * 0.0125;
        end

        return schedule_lr
    end
end

# LR Scheduler with warm-up
mutable struct StateScheduler
    # record
    current_iteration::Int64    # progressed iteration 
    current_lr::Float64         # current scheduled learning rate
    current_epoch::Int16        # progressed training epoch
    
    warmup_epoch::Int16         # as the state switch indicator
    steady_epoch::Int16

    warmup_iteration::Int32     # 
    decayFn::Any                # pointer to ParameterSchedulers
    steadyFn::Any               # pointer to ParameterSchedulers
    base_lr::Float32            # base learning rate
    target_epoch::Int16         # target training epoch    
end

@functor StateScheduler
# constructor
function StateScheduler(target_epoch=1, base_lr=1, warmup_iter=1)
    decayFn = Inv(λ=base_lr, p=2, γ=0.0433)
    steadyFn = SinDecay2(λ0=base_lr, λ1=base_lr * 1.02, period=4)
    StateScheduler(0, 0, 0, 0, 0, warmup_iter, decayFn, steadyFn, base_lr, target_epoch)
end

function (x::StateScheduler)(iteration_no, epoch_no)
    x.current_iteration = iteration_no
    x.current_epoch = epoch_no

    if x.current_iteration <= x.warmup_iteration
        # warmup 
        x.current_lr = x.base_lr * ((x.current_iteration / x.warmup_iteration)^4);
        x.warmup_epoch = x.current_epoch;
        
    elseif x.current_iteration >= x.warmup_iteration && 
        epoch_no < (x.warmup_epoch + floor(0.33 * (x.target_epoch - x.warmup_epoch))) 
        # constant learning rate for 33% of target_epoch
        # x.current_lr = x.base_lr;
        x.current_lr = x.steadyFn(x.current_epoch - x.warmup_epoch + 4)
        x.steady_epoch = x.current_epoch        
    else
        # Decay learning rate
        x.current_lr = x.decayFn(x.current_epoch - x.steady_epoch);
    end

    return x.current_lr
end

## Dataset Parameter
# --Init Dataset
NDS.init();
# -- specify training data fold
training_data = NDS.dataFold_5.training_data
# -- dispatching parameter
max_data_iter = training_data.n
data_dispatch_size = 2

## Network Parameter
input_size = [224 224] # W x H x 3Channel

# Training Parameters 
no_epoch = 1
learning_rate = 0.0077
momentum_term = 0.9
warmupPeriod = 3700 # 2800@45

# Optimizer
opt = Momentum(learning_rate, momentum_term) # SGD w/ Momentum

# scheduler
schlr = StateScheduler(no_epoch, learning_rate, warmupPeriod)

# custom loss enable
debug_gradsection_enable = false

## specify model
model = model_1;

## Training Loop
training_loss = 0
epoch_count = 2
iteration_count = 1

# loss values
box_loss = 0f0
obj_loss = 0f0
pxl_loss = 0f0
cus_loss = 0f0
# loss by output branch
b1_boxloss = 0f0
b2_boxloss = 0f0
b1_objloss = 0f0
b2_objloss = 0f0

## Training Loop
@epochs no_epoch begin
    # reset datastore
    NDS.resetDispatchRecord();
    current_data_iter = NDS.getDispatchRecord();   

    progmeter = Progress(max_data_iter - data_dispatch_size + 1, 1)
 
    while current_data_iter <= (max_data_iter - data_dispatch_size + 1)
        
        update!(progmeter, current_data_iter)
        # load data (output as Flux.Data.DataLoader)
        trainingDL = NDS.dispatchData(training_data;
            dispatch_size=data_dispatch_size,
            shuffle_enable=false,
            img_outputsize=[16 16]
        );

        # update datastore index
        current_data_iter = NDS.getDispatchRecord()


        # update learning rate       
        # used_lr = scheduleLR(iteration_count, epoch_count, learning_rate, warmupPeriod, no_epoch) 
        # global opt.eta = used_lr;
        global opt.eta = schlr(iteration_count, epoch_count)

        # update iteration counter
        global iteration_count += 1;

        # Log Training Loss in iteration
        if log_enable
            with_logger(logger) do
                @info "train/total" training_loss
                @info "train/box" box_loss
                @info "train/obj" obj_loss
                @info "train/pxl" pxl_loss
                @info "train/cus" cus_loss
                @info "branch1/box" b1_boxloss
                @info "branch1/obj" b1_objloss
                @info "branch2/box" b2_boxloss
                @info "branch2/obj" b2_objloss
                @info "LR" opt.eta
            end
        end

        @show current_data_iter
        @show iteration_count

    end # iteration

    @show training_loss
    # update epoch counter
    global epoch_count += 1;

end # epoch    