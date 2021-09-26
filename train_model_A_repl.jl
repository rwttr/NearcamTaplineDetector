# TrainingA.jl
# Custom Training Loop with NearcamTaplineDataset Framework
# update from train3.jl
# - add support for parallel supervising layers
# - remove all tapline ss customloss

""" Dependencies and Function Definition """
##
import Pkg
Pkg.activate(".")
Pkg.instantiate()

include("model.jl")

# Network core
using Flux
using Flux:Data.DataLoader
using Flux:@epochs
using CUDA

# Utils
using Dates
using Statistics
using ImageView
using Logging
using TensorBoardLogger
using BSON:@save
using BSON:@load
using ParameterSchedulers
using Functors
using ProgressMeter

using Augmentor
using Random

LinearAlgebra.BLAS.set_num_threads(4)

""" Learning Rate Scheduler with warm-up and decay """
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
        x.current_lr = x.steadyFn(x.current_epoch - x.warmup_epoch + 4)
        x.steady_epoch = x.current_epoch
    else
        # Decay learning rate
        x.current_lr = x.decayFn(x.current_epoch - x.steady_epoch);
    end

    return x.current_lr
end

# Augmentation : image, bbox, pxlmask 
function augmentTapline(img::Array{Float32,4}, bbox::Array{Float32,4}, pxlmask::Array{Float32,4})
    max_x, max_y, _, batchsize = size(img)

    img_out = similar(img)
    pxl_out = similar(pxlmask)
    box_out = similar(bbox)

    rot_range = -5:0.25:5
    zoom_range = 0.9:0.02:1.1
    
    Threads.@threads for batch_no = 1:batchsize
        rot_angle = rand(rot_range)
        zoom_factor = rand(zoom_range)

        img_batchdim = (img[:,:,1,batch_no], img[:,:,2,batch_no], img[:,:,3,batch_no]);        
        pxl_batchdim = pxlmask[:,:,1,batch_no];

        # transformation pipeline
        pl = Either(Rotate(rot_angle), NoOp()) |> Resize(max_x, max_y) |> 
        Either(Zoom(zoom_factor), NoOp())

        img_p = Augmentor.unwrap.(augment((img_batchdim..., Augmentor.Mask(pxl_batchdim)), pl));
        img_out[:,:,:,batch_no] = cat(img_p[1], img_p[2], img_p[3], dims=3);
        
        # recalculate bbox and mask
        warp_pxl = map(img_p[4]) do x
            if x > 0
                x = 1f0  
            else
                x = 0f0            
            end
        end

        # box 
        warp_pxl_bin = Bool.(warp_pxl);
        ep_1 = findfirst(warp_pxl_bin)
        ep_2 = findlast(warp_pxl_bin)
        y1, x1 = ep_1.I        
        y2, x2 = ep_2.I
        
        # x = x1
        # y = y1
        # w = x2 - x1
        # h = y2 - y1
        box_out[:,:,1,batch_no] = Float32.([x1 y1 x2 - x1 y2 - y1])

        # mask
        warp_pxlmask_bin = zeros(Bool, max_x, max_y)
        active_px = argmax(img_p[4], dims=1)
        active_px = filter(active_px) do x
            x.I[2] in x1:x2
        end

        warp_pxlmask_bin[active_px[:]] .= 1

        pxl_out[:,:,:,batch_no] = Float32.(warp_pxlmask_bin)

        # debug
        # NDS.showImageSample(img_out[:,:,:,batch_no], box_out[:,:,1,batch_no])
        # NDS.showImageSample(pxl_out[:,:,:,batch_no], box_out[:,:,1,batch_no])
        # imshow(0.5 * img_out[:,:,1,batch_no] .+ 0.5 * pxl_out[:,:,:,batch_no])
        # imshow(0.5 * img_p[1] .+ 0.5 * img_p[2])
    end

    return img_out, box_out, pxl_out
end


##
""" Config Section """

# datafold selection
fold_no = 5

# specify model from model.jl
model_name = "model_A"
model = model_a;

# save / model checkpoint enable
chkpoint_enable = true

# resume training
resume_training = false
resume_epoch = 67

# gpu select
gpu_enable = true

# tensorboard enable
tensorboard_enable = true
if tensorboard_enable
    if resume_training
        logger = TBLogger("tensorboardlog/$model_name/fold$fold_no/loss", tb_append);
    else
        logger = TBLogger("tensorboardlog/$model_name/fold$fold_no/loss", tb_overwrite);
    end
    # tensorboard_cmd = `tensorboard --logdir=tensorboardlog`
    # tensorboard_cmd = `tensorboard --logdir=tensorboardlog/$model_name/fold$fold_no`
end

# Training Parameters
no_epoch = 100                  # max training epoch
learning_rate = 0.008           # base learning rate
momentum_term = 0.9
warmupPeriod = 4000             # 2800@45
iou_penalty_threshold = 0.8     # prediction iou to ignore loss calculate (yolo branch)



""" NN Training Section """

target_epoch = no_epoch
checkpoint_epoch = 1:target_epoch

# scheduler
schlr = StateScheduler(no_epoch, learning_rate, warmupPeriod)

# training loop record
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

# Init Dataset
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
# Network Parameter
input_size = [224 224] # W x H x 3 Channel
max_epoch_iter = max_data_iter - data_dispatch_size + 1



""" Training Loop section """
# Optimizer
opt = Momentum(learning_rate, momentum_term) # SGD w/ Momentum
ps = params(model)

# loss values
total_loss = 0f0 # total loss

# yolo loss
total_box_loss = 0f0
total_obj_loss = 0f0
# yolo loss by output branch
b1_boxloss = 0f0
b2_boxloss = 0f0
b1_objloss = 0f0
b2_objloss = 0f0

# pxl branch supervisioning loss
pxl_loss_1 = 0f0
pxl_loss_2 = 0f0
pxl_loss_3 = 0f0
total_pxl_loss = 0f0

## Training Loop
@epochs no_epoch begin
    # reset datastore
    NDS.resetDispatchRecord();
    current_data_iter = NDS.getDispatchRecord();
    progmeter = Progress(max_epoch_iter, showspeed=true)
    global epoch_count += 1;

    while current_data_iter <= max_epoch_iter
        # update progress bar
        update!(progmeter, current_data_iter)
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
        target_yolo_b1 = generateYolotarget(gtbox, yololayer_layer1) # yolohead
        target_yolo_b2 = generateYolotarget(gtbox, yololayer_layer2) # yolohead
        target_pxl = gtpxl; # pxl-head
        if gpu_enable
            img = gpu(img)
            target_yolo_b1 = gpu(target_yolo_b1)
            target_yolo_b2 = gpu(target_yolo_b2)
            target_pxl = gpu(target_pxl)
        end

        # compute loss and network gradient
        g = Flux.gradient(ps) do
            # forward data to network
            nnResponse = model(img)
            response_yolo_b1 = nnResponse[1] # Yolobranch_1
            response_yolo_b2 = nnResponse[2] # Yolobranch_2
            # index 3 reserve for inference
            response_pxl_1 = nnResponse[4] # LossPxl_1
            response_pxl_2 = nnResponse[5] # LossPxl_2
            response_pxl_3 = nnResponse[6] # LossPxl_3

            # this section mutate arrays
            # check outputs to groundtruth for loss penalties
            local iou_mask_layer1
            local iou_mask_layer2
            Zygote.ignore() do
                iou_layer1 = predBoxOverlap(response_yolo_b1 |> cpu, gtbox, yololayer_layer1)
                iou_layer2 = predBoxOverlap(response_yolo_b2 |> cpu, gtbox, yololayer_layer2)

                iou_mask_layer1 = maskPredBoxOverlap(iou_layer1)
                iou_mask_layer2 = maskPredBoxOverlap(iou_layer2)

                if gpu_enable
                    iou_mask_layer1 = gpu(iou_mask_layer1)
                    iou_mask_layer2 = gpu(iou_mask_layer2)
                end
            end

            # calculate loss
            # box loss
            lossBox_1 = yoloBoxLoss(response_yolo_b1, target_yolo_b1, iou_mask_layer1)
            lossBox_2 = yoloBoxLoss(response_yolo_b2, target_yolo_b2, iou_mask_layer2)

            # objectness loss
            lossObj_1 = yoloObjLoss(response_yolo_b1, target_yolo_b1, iou_mask_layer1)
            lossObj_2 = yoloObjLoss(response_yolo_b2, target_yolo_b2, iou_mask_layer2)

            # segmentation loss            
            lossPxl_1 = pxlLoss_tversky(response_pxl_1, target_pxl)            
            lossPxl_2 = pxlLoss_dice(response_pxl_2, target_pxl)
            lossPxl_3 = pxlLoss_binarycrossentropy(response_pxl_3, target_pxl)
                    
            loss = lossBox_1 + lossObj_1 + lossBox_2 + lossObj_2 + lossPxl_1 + lossPxl_2 + lossPxl_3;
            
            # log global loss vaiables (for TensorBoardLogger)
            Zygote.ignore() do                
                if tensorboard_enable
                    global total_loss = loss;
                    # yolo
                    global total_obj_loss = lossObj_1 + lossObj_2;
                    global total_box_loss = lossBox_1 + lossBox_2;
                    global b1_boxloss = lossBox_1;
                    global b2_boxloss = lossBox_2;
                    global b1_objloss = lossObj_1;
                    global b2_objloss = lossObj_2;
                    # pxl
                    global pxl_loss_1 = lossPxl_1;
                    global pxl_loss_2 = lossPxl_2;
                    global pxl_loss_3 = lossPxl_3;
                    global total_pxl_loss = pxl_loss_1 + pxl_loss_2 + pxl_loss_3
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
                @info "train/total" total_loss
                @info "train/box" total_box_loss
                @info "train/obj" total_obj_loss
                @info "train/pxl" total_pxl_loss

                @info "b1/box" b1_boxloss
                @info "b1/obj" b1_objloss
                @info "b2/box" b2_boxloss
                @info "b2/obj" b2_objloss

                @info "pxl/LossPxl_1" pxl_loss_1
                @info "pxl/LossPxl_2" pxl_loss_2
                @info "pxl/LossPxl_3" pxl_loss_3

                @info "LR" opt.eta
            end
        end
    end # iteration loop

    @show Dates.now();
    @show iteration_count
    if tensorboard_enable
        with_logger(logger) do
            @info "train/epoch_loss" total_loss
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
