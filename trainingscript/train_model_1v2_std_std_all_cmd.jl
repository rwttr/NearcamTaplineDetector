import Pkg
Pkg.activate(".")
Pkg.instantiate()

""" command line argument parsing section """

using ArgParse

function parse_cmd()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--fold_no"
            help = "specify training datafold"
            arg_type = Int
            default = 0
        "--gpu_id"
            help = "specify gpu id"
            arg_type = Int
            default = 0
    end
    return parse_args(s)
end

p_args = parse_cmd()

## 
# Custom Training Loop with NearcamTaplineDataset Framework

""" Dependencies """

# Network core
using Flux
using Flux:Data.DataLoader
using Flux:@epochs
using CUDA

# Utils
using Dates
using Statistics
# using ImageView
using Logging
using TensorBoardLogger
using BSON:@save
using BSON:@load
using ParameterSchedulers
using Functors
using ProgressMeter
using Augmentor
using Random

# LinearAlgebra.BLAS.set_num_threads(4)
include("model.jl")

##
"""" Config Section """

# datafold selection
fold_no = Int(p_args["fold_no"])

## specify model
model_name = "model_1v2_std_std_all"
model = model_1v2;

# save / model checkpoint enable
chkpoint_enable = true

# resume training
resume_training = false
resume_epoch = 0

# gpu select
gpu_enable = true
CUDA.device!(p_args["gpu_id"])

# tensorboard enable
tensorboard_enable = true
if tensorboard_enable
    if resume_training
        logger = TBLogger("tensorboardlog/$model_name/fold$fold_no/loss", tb_append);
    else
        logger = TBLogger("tensorboardlog/$model_name/fold$fold_no/loss", tb_overwrite);
    end
    # tensorboard_cmd = `tensorboard --logdir=content`
    # tensorboard_cmd = `tensorboard --logdir=content/$model_name/fold$fold_no`
end

# Training Parameters
no_epoch = 100
learning_rate = 0.008           # base learning rate
momentum_term = 0.9
warmupPeriod = 4000             # 2800@45
iou_penalty_threshold = 1       # prediction iou to ignore loss calculate (yolo branch)



""" NN Training Section """

target_epoch = no_epoch
# checkpoint_epoch = 1:9:target_epoch
checkpoint_epoch = [99 100]

# scheduler
schlr = StateScheduler(no_epoch, learning_rate, warmupPeriod)

# traning record
epoch_count = 0
iteration_count = 0

if resume_training
    local model_save
    local schlr_save
    local logger_save
    resume_model_path = "weights/" * "$model_name/fold$fold_no" * "/$model_name" * "_fold$fold_no" * "_epoch$resume_epoch.bson"
    resume_schlr_path = "weights/" * "$model_name/fold$fold_no" * "/schlr_$model_name" * "_fold$fold_no" * "_epoch$resume_epoch.bson"
    logger_path =  "weights/" * "$model_name/fold$fold_no" * "/loggercounter_$model_name" * "_fold$fold_no" * "_epoch$resume_epoch.bson"
    # load model
    @load resume_model_path model_save
    model = model_save;

    # load current record
    @load resume_schlr_path schlr_save
    schlr = schlr_save

    epoch_count = schlr.current_epoch # recorded epoch
    iteration_count = schlr.current_iteration # recorded iteration_count
    no_epoch = schlr.target_epoch - schlr.current_epoch  # epoch to go

    if tensorboard_enable
        @load logger_path logger_save
        logger_offset = logger_save
        # set tensorboard logger offset
        set_step!(logger, logger_offset)
    end
end

if gpu_enable
    model = model |> gpu;
end

## Init Dataset
NDS.init();
# -- specify training data fold
fold_table = [
    NDS.dataFold_1.training_data;
    NDS.dataFold_2.training_data;
    NDS.dataFold_3.training_data;
    NDS.dataFold_4.training_data;
    NDS.dataFold_5.training_data;
];

training_data = fold_table[fold_no]
max_data_iter = training_data.n
data_dispatch_size = 2
## Network Parameter
input_size = [224 224] # W x H x 3Channel
max_epoch_iter = max_data_iter - data_dispatch_size + 1



""" Training Loop section """

## Training Loop
# Optimizer
opt = Momentum(learning_rate, momentum_term) # SGD w/ Momentum
ps = params(model)

# loss values
training_loss = 0f0

box_loss = 0f0
obj_loss = 0f0
pxl_loss = 0f0

# loss by output branch
b1_boxloss = 0f0
b2_boxloss = 0f0
b1_objloss = 0f0
b2_objloss = 0f0

# pixel loss
pxl_loss_1 = 0f0
pxl_loss_2 = 0f0
pxl_loss_3 = 0f0

# training loop
@epochs no_epoch begin
    # reset datastore
    NDS.resetDispatchRecord();
    current_data_iter = NDS.getDispatchRecord();
    # progmeter = Progress(max_epoch_iter, showspeed=true)
    global epoch_count += 1;

    while current_data_iter <= max_epoch_iter
        # update progress bar
        # update!(progmeter, current_data_iter)
        # update iteration counter
        global iteration_count += 1;
        # update epoch counter

        # load data (output as Flux.Data.DataLoader)
        trainingDL = NDS.dispatchData(training_data;
            dispatch_size=data_dispatch_size,
            shuffle_enable=true,
            img_outputsize=input_size
        );

        # update datastore index
        current_data_iter = NDS.getDispatchRecord()

        # input Image and groundtruth
        img_ = Float32.(trainingDL.data[1]) |> normalizeImg # multi-thread        
        gtbox_ = Float32.(trainingDL.data[2])   # bbox
        gtpxl_ = Float32.(trainingDL.data[3])   # pixel label

        # augmented data with rotation and scaling
        img, gtbox, gtpxl = augmentTapline(img_, gtbox_, gtpxl_);

        # generate network target
        targetfor_branch1 = generateYolotarget(gtbox, yololayer_layer1) # yolohead
        targetfor_branch2 = generateYolotarget(gtbox, yololayer_layer2) # yolohead
        targetfor_branch3 = gtpxl; # pxl-head
        if gpu_enable
            img = gpu(img)
            targetfor_branch1 = gpu(targetfor_branch1)
            targetfor_branch2 = gpu(targetfor_branch2)
            targetfor_branch3 = gpu(targetfor_branch3)
        end

        # compute loss and network gradient
        g = Flux.gradient(ps) do
            # forward data to network
            nnResponse = model(img)
            response_branch1 = nnResponse[1]
            response_branch2 = nnResponse[2]
            response_branch3 = nnResponse[3]

            # this section mutate arrays
            # check outputs to groundtruth for loss penalties
            local iou_mask_layer1
            local iou_mask_layer2
            Zygote.ignore() do
                iou_layer1 = predBoxOverlap(response_branch1 |> cpu, gtbox, yololayer_layer1)
                iou_layer2 = predBoxOverlap(response_branch2 |> cpu, gtbox, yololayer_layer2)

                iou_mask_layer1 = maskPredBoxOverlap(iou_layer1, mask_iou_threshold=iou_penalty_threshold)
                iou_mask_layer2 = maskPredBoxOverlap(iou_layer2, mask_iou_threshold=iou_penalty_threshold)

                if gpu_enable
                    iou_mask_layer1 = gpu(iou_mask_layer1)
                    iou_mask_layer2 = gpu(iou_mask_layer2)
                end
            end

            # calculate loss
            # box loss
            lossBox_1 = yoloBoxLoss(response_branch1, targetfor_branch1, iou_mask_layer1)
            lossBox_2 = yoloBoxLoss(response_branch2, targetfor_branch2, iou_mask_layer2)

            # objectness loss
            lossObj_1 = yoloObjLoss(response_branch1, targetfor_branch1, iou_mask_layer1)
            lossObj_2 = yoloObjLoss(response_branch2, targetfor_branch2, iou_mask_layer2)

            # segmentation loss            
            lossPxl_1 = pxlLoss_tversky(response_branch3, targetfor_branch3)            
            lossPxl_2 = pxlLoss_dice(response_branch3, targetfor_branch3)
            lossPxl_3 = pxlLoss_focal(response_branch3, targetfor_branch3)
            lossPxl = lossPxl_1 + lossPxl_2 + lossPxl_3

            loss = lossBox_1 + lossObj_1 + lossBox_2 + lossObj_2 + lossPxl;

            Zygote.ignore() do
                # plot loss on TensorBoardLogger
                if tensorboard_enable
                    global training_loss = loss;
                    global obj_loss = lossObj_1 + lossObj_2;
                    global box_loss = lossBox_1 + lossBox_2;
                    global pxl_loss = lossPxl;
                    global b1_boxloss = lossBox_1;
                    global b2_boxloss = lossBox_2;
                    global b1_objloss = lossObj_1;
                    global b2_objloss = lossObj_2;

                    global pxl_loss_1 = lossPxl_1;
                    global pxl_loss_2 = lossPxl_2;
                    global pxl_loss_3 = lossPxl_3;
                end
            end

            return loss;
        end

        # update learning rate
        global opt.eta = schlr(iteration_count, epoch_count)

        # update weights
        Flux.update!(opt, ps, g);

        # Log Training Loss in iteration
        if tensorboard_enable
            with_logger(logger) do
                @info "train/total" training_loss
                @info "train/box" box_loss
                @info "train/obj" obj_loss
                @info "train/pxl" pxl_loss

                @info "b1/box" b1_boxloss
                @info "b1/obj" b1_objloss
                @info "b2/box" b2_boxloss
                @info "b2/obj" b2_objloss

                @info "pxl/LossPxl_1" pxl_loss_1
                @info "pxl/LossPxl_2" pxl_loss_2
                @info "pxl/LossPxl_3" pxl_loss_3
                @info "pxl/total" pxl_loss

                @info "LR" opt.eta
            end
        end
    end # iteration loop

    # @show Dates.now();
    # @show iteration_count
    if tensorboard_enable
        with_logger(logger) do
            @info "train/epoch_loss" training_loss
        end
    end

    # checkpoint
    if chkpoint_enable && (epoch_count in checkpoint_epoch)
        # move model to cpu before save
        model_save = cpu(model)
        # weights = params(model_save)
        # @save "weights/weight_model_1_epoch$epoch_count.bson" weights
        @save "weights/" * "$model_name/fold$fold_no" * "/$model_name" * "_fold$fold_no" * "_epoch$epoch_count.bson" model_save

        # save scheduler
        schlr_save  = schlr
        @save "weights/" * "$model_name/fold$fold_no" * "/schlr_$model_name" * "_fold$fold_no" * "_epoch$epoch_count.bson" schlr_save

        # save TensorBoardLogger iteration count
        if tensorboard_enable
            logger_save = TensorBoardLogger.step(logger)
            @save "weights/" * "$model_name/fold$fold_no" * "/loggercounter_$model_name" * "_fold$fold_no" * "_epoch$epoch_count.bson" logger_save
        end
    end

end # epoch

println("training done on fold_no:$fold_no")
exit()