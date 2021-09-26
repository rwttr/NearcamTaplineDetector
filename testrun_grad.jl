# Training.jl
# Custom Training Loop with NearcamTaplineDataset Framework

## load dataset
include("model.jl")
include("NearcamTaplineDataset.jl");
import .NearcamTaplineDataset as NDS;

# Network core
using Flux
using Flux:Data.DataLoader
using Flux:@epochs
using CUDA

# Utils
using Statistics
using ImageView
import Zygote

# limit blas thread for performance
LinearAlgebra.BLAS.set_num_threads(4)

## Dataset Parameter
# --Init Dataset
NDS.init();
# -- specify training data fold
training_data = NDS.dataFold_1.training_data;
# -- dispatching parameter
max_data_iter = training_data.n;
data_dispatch_size = 2

## Network Parameter
input_size = [224 224] # W x H x 3Channel

# Training Parameters 
no_epoch = 100
learning_rate = 0.0075
momentum_term = 0.9
warmupPeriod = 3800 # 2800@45
iou_penalty_threshold = 0.5

# gpu
gpu_enable = true

customloss_enable = false
custom_loss_enable2 = false
custom_loss_enable3 = false
custom_loss_enable4 = true

## specify model
model = f32(model_1);

## Training Loop
training_loss = 0
epoch_count = 1
iteration_count = 1
if gpu_enable
    model = model |> gpu;
end
ps = params(model); # Zygote params

# Optimizer
opt = Momentum(learning_rate, momentum_term); # SGD w/ Momentum

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

# load data (output as Flux.Data.DataLoader)
trainingDL = NDS.dispatchData(
    training_data; dispatch_size=data_dispatch_size,shuffle_enable=true,img_outputsize=input_size
);

# update datastore index
current_data_iter = NDS.getDispatchRecord()

# input Image
img = Float32.(trainingDL.data[1]) |> normalizeImg # multi-thread
# groundtruth
gtbox = Float32.(trainingDL.data[2])   # bbox
gtpxl = Float32.(trainingDL.data[3])   # pixel label

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

Flux.@epochs 4 begin
    @time g = Flux.gradient(ps) do
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

            iou_mask_layer1 = maskPredBoxOverlap(iou_layer1)
            iou_mask_layer2 = maskPredBoxOverlap(iou_layer2)

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
        # lossPxl = pxlLoss_tversky(response_branch3, targetfor_branch3)
        lossPxl = pxlLoss_dice(response_branch3, targetfor_branch3)
        # lossPxl = wbcrossentropy(response_branch3, targetfor_branch3)

        ss_loss = 0f0
        if customloss_enable
            # targetfor_branch1_cpu = cpu(targetfor_branch1)
            # targetfor_branch2_cpu = cpu(targetfor_branch2) 
            targetfor_branch3_cpu = gtpxl
  
            # transform t- to b-
            trans_response_b1 = transYolohead(response_branch1, yololayer_layer1)
            trans_response_b2 = transYolohead(response_branch2, yololayer_layer2)

            # trans_response_b1_cpu = cpu(trans_response_b1)
            # trans_response_b2_cpu = cpu(trans_response_b2)
            response_branch3_cpu = cpu(response_branch3)

            # locate b-values at gt-objectness
            # pred_box1 = predictYoloBoxTrain(trans_response_b1_cpu, targetfor_branch1_cpu)
            # pred_box2 = predictYoloBoxTrain(trans_response_b2_cpu, targetfor_branch2_cpu)

            # gpu test
            pred_box1 = predictYoloBoxTrain(trans_response_b1, targetfor_branch1)
            pred_box2 = predictYoloBoxTrain(trans_response_b2, targetfor_branch2)
            pred_box1 = cpu(pred_box1)
            pred_box2 = cpu(pred_box2)

            tapline_1 = predictTaplinePoints(response_branch3_cpu, pred_box1)
            tapline_2 = predictTaplinePoints(response_branch3_cpu, pred_box2)

            ss_loss1 = taplineSSLoss(tapline_1, targetfor_branch3_cpu)
            ss_loss2 = taplineSSLoss(tapline_2, targetfor_branch3_cpu) 
            
            ss_loss = ss_loss1 + ss_loss2
            if gpu_enable
                ss_loss = gpu(ss_loss)
            end

        end

        if custom_loss_enable2
            # crop tapline edgemap with gtbox
            targetfor_branch3_cpu = gtpxl
            response_branch3_cpu = cpu(response_branch3)

            # compute loss inside mask
            tapline_map = cropTaplinePoints(response_branch3_cpu, gtbox)
            ss_loss = taplineSSLoss(tapline_map, targetfor_branch3_cpu)

            if gpu_enable
                ss_loss = gpu(ss_loss)
            end
        end

        if custom_loss_enable3
            #targetfor_branch3_cpu = gtpxl
            response_branch3_cpu = cpu(response_branch3)
            ss_loss = straightnessLoss4(gtpxl, response_branch3_cpu, gtbox)
            if gpu_enable
                ss_loss = gpu(ss_loss)
            end
        end

        if custom_loss_enable4
            ss_loss = straightnessLoss5(targetfor_branch3, response_branch3)
        end

        loss = lossBox_1 + lossObj_1 + lossBox_2 + lossObj_2 + lossPxl + ss_loss;
        return loss;
    end
    Flux.update!(opt, ps, g);
end


