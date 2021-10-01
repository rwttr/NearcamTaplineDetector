## load dependencies
using Zygote:Array
using Flux: batch, Zygote
using CUDA:findmax
using Statistics:LinearAlgebra
using StatsBase:isempty

using BSON:@load
using BSON:@save
using ImageView
using Images
using CUDA

include("model.jl")

include("NearcamTaplineDataset.jl")
import .NearcamTaplineDataset as NDS
NDS.init();
data_dispatch_size = 1;
nn_inputsize = [224 224];       # network input size

## Load Model
model_path = "weights/model_1v1_std_std_dice/fold1/model_1v1_dice_fold1_epoch100.bson"

@load model_path model_save
model = model_save;

gpu_enable = true;
if gpu_enable
    model = gpu(model);
end

""" Sample Detection """
## Show sample image
testdata = NDS.dataFold_1.test_data;
NDS.resetDispatchRecord();

testDL = NDS.dispatchData(testdata;
    dispatch_size=data_dispatch_size,
    shuffle_enable=false,
    img_outputsize=nn_inputsize
);

img = Float32.(testDL.data[1]) |> normalizeImg;
gtbox = Float32.(testDL.data[2]);
gtpxl = Float32.(testDL.data[3]);
NDS.showImageSample(img[:,:,:,1],gtbox[:,:,1,1]);

if gpu_enable
    img = gpu(img);
end;

## run single detection
nnoutput = model(img)

nnoutput = nnoutput |> cpu;
output_branch1 = nnoutput[1];
output_branch2 = nnoutput[2];
output_branch3 = nnoutput[3];

# predict boxes from yolohead 
#    output in format Vector{Vector{T} where T} x 1 batchsize, [1] extract to Vector{}
pred_box1 = predictBox(output_branch1, yololayer_layer1, obj_th=0.0)[1]
pred_box2 = predictBox(output_branch2, yololayer_layer2, obj_th=0.0)[1]

# gather all box prediction & non-max 
pred_box = vcat(pred_box1, pred_box2) 
pred_box = nmsBoxwScore([pred_box])[1]

# pick one box with highest score
pred_box_score = map(x -> x[5], pred_box)
pred_box_max_idx = findmax(pred_box_score)[2]
pred_box = pred_box[pred_box_max_idx]

# argmax position on edgemap -> tapping line within bbox
taplineimg = predictTapline(output_branch3, [[pred_box]])

dtbox = pred_box
dttapline = taplineimg[][]
sourceimg = img[:,:,:,1] |> cpu;

## show detection 
NDS.showImageSample(dttapline, dtbox);
NDS.showImageSample(sourceimg, dtbox);

figplot = ImageView.imshow(sourceimg);

taplinepoints = findall(Bool.(dttapline[:,:,1]))
# convert to Vector{Tuple{x,y)}}
taplinepoints = map(x -> x.I, taplinepoints)
# swap coordinate
taplinepoints = map(x -> (x[2], x[1]), taplinepoints)

ImageView.annotate!(
    figplot,
    ImageView.AnnotationPoints(taplinepoints, size=2, shape='.', color=RGB(1, 0.1, 0))
); # left top right bottom

xb, yb, wb, hb = dtbox[1:4]
ImageView.annotate!(
    figplot,
    ImageView.AnnotationBox(xb, yb, xb + wb, yb + hb, linewidth=2, color=Images.RGB(0, 1, 0))
); # left top right bottom




## Detection loop 
""" Detection Section """
fold_no = 4                             # specify test datafold
gpu_enable = true

# Load Model
model_path = "weights/model_A/fold4/model_A_fold4_epoch100.bson"

@load model_path model_save
model = model_save;

gpu_enable = true;
if gpu_enable
    model = gpu(model);
end

testdata = [
    NDS.dataFold_1.test_data    
    NDS.dataFold_2.test_data
    NDS.dataFold_3.test_data
    NDS.dataFold_4.test_data
    NDS.dataFold_5.test_data
];
testdata = testdata[fold_no]

# data dispatch params
data_dispatch_size = 1;         # batchsize
nn_inputsize = [224 224];       # network input size

NDS.init()                      # build all datafold
NDS.resetDispatchRecord()
current_data_iter = NDS.getDispatchRecord()
max_data_iter = testdata.n

# init results
dtboxiou_array = zeros(testdata.n)
dttapline_array = zeros(Bool, nn_inputsize..., testdata.n)

gttapline_array = zeros(Bool, nn_inputsize..., testdata.n)

counter_i = 1
while current_data_iter <= (max_data_iter - data_dispatch_size + 1)

    # dataloader
    testDL = NDS.dispatchData(testdata; 
        dispatch_size=data_dispatch_size, 
        shuffle_enable=false,
        img_outputsize=nn_inputsize
    );

    # update datastore index
    current_data_iter = NDS.getDispatchRecord()  

    # image data
    img = Float32.(testDL.data[1]) |> normalizeImg;
    # groundtruth
    gtbox = Float32.(testDL.data[2]);    
    gtpxl = Float32.(testDL.data[3]);
    
    if gpu_enable
        img = gpu(img);
    end

    # model inference
    nnoutput = model(img)
    # push all results to cpu / all evaluation done on cpu
    nnoutput = nnoutput |> cpu;
    output_branch1 = nnoutput[1];
    output_branch2 = nnoutput[2];
    output_branch3 = nnoutput[3];

    pred_box1 = predictBox(output_branch1, yololayer_layer1, obj_th=0.0)[1]
    pred_box2 = predictBox(output_branch2, yololayer_layer2, obj_th=0.0)[1]

    # gather all box prediction & non-max 
    pred_box = vcat(pred_box1, pred_box2) 
    pred_box = nmsBoxwScore([pred_box])[1]

    # pick one box with highest score
    pred_box_score = map(x -> x[5], pred_box)
    pred_box_max_idx = findmax(pred_box_score)[2]
    pred_box = pred_box[pred_box_max_idx]

    # argmax position on edgemap -> tapping line within bbox
    taplineimg = predictTapline(output_branch3, [[pred_box]])

    dtbox = pred_box
    dttapline = taplineimg[][]

    iou = bboxIoU(dtbox, gtbox[:])

    dtboxiou_array[counter_i] = iou
    dttapline_array[:,:,counter_i] = Bool.(dttapline)
    gttapline_array[:,:,counter_i] = Bool.(gtpxl)

    # update counter
    counter_i += 1
end

## Save result
dtboxiou = dtboxiou_array;
dttaplineimg = dttapline_array

@save "result/model_A/dtboxiou_fold4.bson" dtboxiou
@save "result/model_A/dttapline_fold4.bson" dttaplineimg

## save groundtruth
# gttapline = gttapline_array;
# @save "result/gttapline_fold1.bson" gttapline
