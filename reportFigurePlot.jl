## 
using Plots
plotlyjs()

# load dependencies
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

# Load Model
# model_path = "weights/no_iou_penalty/model_a/fold1/model_a_fold1_epoch99.bson"
# model_path = "weights/model_a_reg/fold1/model_a_reg_fold1_epoch100.bson"
# model_path = "weights/model_a2_reg/fold1/model_a2_reg_fold1_epoch100.bson"

# model_path = "weights/model_b_dice/fold1/model_b_dice_fold1_epoch100.bson"
# model_path = "weights/model_c_dice/fold1/model_c_dice_fold1_epoch100.bson"

model_path = "weights/model_b_focal/fold1/model_b_focal_fold1_epoch100.bson"
model_path = "weights/model_c_focal/fold1/model_c_focal_fold1_epoch100.bson"

@load model_path model_save
model = model_save;
model = gpu(model);


""" Sample Detection """
## select dataFold
testdata = NDS.dataFold_1.test_data;
NDS.resetDispatchRecord();

## Start Dispatch Data
testDL = NDS.dispatchData(testdata;
    dispatch_size=data_dispatch_size,
    shuffle_enable=false,
    img_outputsize=nn_inputsize
);

img = Float32.(testDL.data[1]) |> normalizeImg;
img_ = Float32.(testDL.data[1]) # non-normalized image for visualizing

gtbox = Float32.(testDL.data[2]);
gtpxl = Float32.(testDL.data[3]);
NDS.showImageSample(img[:,:,:,1],gtbox[:,:,1,1]);

img = gpu(img);

## run single detection
nnoutput = model(img);

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



## Plot
# base image from dataset load on detect.jl
baseimg = convert2RGB(img_[:,:,:,1] |> cpu) |> copy

# box shape
rectangle(x,y,w,h) = Shape(x .+ [0, w, w, 0], y .+ [0,0,h,h])

gtboxplot = rectangle(gtbox[:,:,1,1]...)
dtboxplot = rectangle(dtbox[1:4]...)

# tapline pixels size(w,h)
# from detect.jl
gtpxlplot = gtpxl[:,:,1,1];
dtpxlplot = dttapline[:,:,1];

gtpxl_px = map(findall(Bool.(gtpxlplot))) do x
    x = (Float32(x.I[2]), Float32(x.I[1]))
end

dtpxl_px = map(findall(Bool.(dtpxlplot))) do x
    x = (Float32(x.I[2]), Float32(x.I[1]))
end

# tapline endpoints
gtpxl_px[1] = (gtbox[:,:,1,1][1], gtbox[:,:,1,1][2])
gtpxl_px[end] = (gtbox[:,:,1,1][1] + gtbox[:,:,1,1][3], gtbox[:,:,1,1][2] + gtbox[:,:,1,1][4])

dtpxl_px[1] = (dtbox[:,:,1,1][1], dtbox[:,:,1,1][2])
dtpxl_px[end] = (dtbox[:,:,1,1][1] + dtbox[:,:,1,1][3], dtbox[:,:,1,1][2] + dtbox[:,:,1,1][4])


# filter dtpxl_px witin dtbox[1:4]
dtpxl_px = filter(dtpxl_px) do x
    a = x
    (a[1] >= dtpxl_px[1][1] && a[1] <= dtpxl_px[end][1]) && 
    (a[2] >= dtpxl_px[1][2] && a[2] <= dtpxl_px[end][2])
end


# 
# start plot
plot(baseimg, size=(400, 400))
# gt-box
plot!(gtboxplot,  
    linewidth=3,
    linestyle=:dot,
    linecolor=:red, 
    fillalpha=0, 
    legend=false
)
# dt-box
plot!(dtboxplot,  
    linewidth=2,
    linecolor=:blue, 
    fillalpha=0, 
    legend=false
)

# tapline
# gt-pxl
plot!(gtpxl_px, 
    markershape=:circle,
    markersize=2,
    markeralpha=0.7,
    markercolor=:red,
    markerstrokewidth=1,
    markerstrokealpha=0.02,
    markerstrokecolor=:red,
    markerstrokestyle=:line,# :dot
    linewidth=2,
    linealpha=0.5,
    linecolor=:red,
)

plot!(dtpxl_px, 
    markershape=:circle,
    markersize=2,
    markeralpha=0.7,
    markercolor=:blue,
    markerstrokewidth=1,
    markerstrokealpha=0.02,
    markerstrokecolor=:blue,
    markerstrokestyle=:line,# :dot
    linewidth=2,
    linecolor=:blue, 
    linealpha=0.5
)


##
# imgA = Bool.(dtpxlplot);
# imgB = Bool.(gtpxlplot);
# @show taplineHausdorffDist(imgA, imgB)
# @show dicef1score(imgA, imgB)

""" Col-Softmax Anaylysis and Visualization """
#data rescale
function rawdatarescale(x)
    temp = x.-minimum(x)
    return temp./(maximum(temp))
end

## plot attention map
# for sample1 - model b (model C for paper)
# const rawimg1 = output_branch3[:,:,1,1];
rawdata1 = rawimg1[:,125];
sum(rawdata1);

rawimg1_gray = convert(Array{N0f8,2},rawimg1) |> rawdatarescale
Gray.(rawimg1_gray) # show in vscode plot pane
plot(Gray.(rawimg1_gray) ,size=(400, 400))



# sample 2 - model_c (model D for paper)
# const rawimg2 = output_branch3[:,:,1,1];
rawdata2 = rawimg2[:,125];
sum(rawdata2);

rawimg2_gray = convert(Array{N0f8,2},rawimg2) |> rawdatarescale
Gray.(rawimg2_gray) # show in vscode plot pane
plot(Gray.(rawimg2_gray) ,size=(400, 400))



""" Column data plot """
col_no1 = 104
col_no2 = 125 

colgt_104 = gtpxl[:,col_no1]
colgt_125 = gtpxl[:,col_no2]

rowgt_104 = Bool.(colgt_104) |> findall
rowgt_125 = Bool.(colgt_125) |> findall

coldata1_104 = rawdatarescale(rawimg1[:,col_no1])
coldata1_125 = rawdatarescale(rawimg1[:,col_no2])

coldata2_104 = rawdatarescale(rawimg2[:,col_no1])
coldata2_125 = rawdatarescale(rawimg2[:,col_no2])

plot(;size=(600,400),
    legend = :bottom,
    xlabel = "Row Number",
    ylabel ="Rescaled Output",
    xlim = (1,230),
    xtick = [1,56,112,168,224],
    ylim = (0,1.01)
)
plot!(coldata1_104, 
    markersize=0,
    markerstrokestyle=:line,# :dot
    linewidth=1.5,
    #linealpha=0.75,
    #linecolor=:yellow,
    labels ="Model C",
    legend = :bottom
)
plot!(coldata2_104, 
    markersize=0,
    markerstrokestyle=:line,# :dot
    linewidth=1.5,
    #linecolor=:red,
    #linealpha=0.75,
    labels ="Model D",
    legend = :bottom
)
vline!(rowgt_104,
    linewidth=1.25,
    linecolor=:black,
    linealpha=0.75,
    linestyle=:dot,
    labels ="Pixel GroundTruth",
)

# col_no2
plot(;size=(600,400),
    legend = :bottom,
    xlabel = "Row Number",
    ylabel ="Rescaled Output",
    xlim = (1,230),
    xtick = [1,56,112,168,224],
    ylim = (0,1.01)
)
plot!(coldata1_125, 
    markersize=0,
    markerstrokestyle=:line,# :dot
    linewidth=1.5,
    #linealpha=0.75,
    #linecolor=:yellow,
    labels ="Model C",
    legend = :bottom
)
plot!(coldata2_125, 
    markersize=0,
    markerstrokestyle=:line,# :dot
    linewidth=1.5,
    #linecolor=:red,
    #linealpha=0.75,
    labels ="Model D",
    legend = :bottom
)
vline!(rowgt_125,
    linewidth=1.25,
    linecolor=:black,
    linealpha=0.75,
    linestyle=:dot,
    labels ="Pixel GroundTruth",
)


# gtbox =  92.225  142.178  64.4  62.5333, col = 143-205

""" Column-line plot on segmentation output map """
# column 104
colline1 = map(repeat([104],outer=224),1:224) do a,b
    (a,b)    
end

# column 125
colline2 = map(repeat([125],outer=224),1:224) do a,b
    (a,b)    
end

plot!(colline1, 
    markersize=0,
    markerstrokestyle=:line,# :dot
    linewidth=1.25,
    linealpha=0.75,
    linecolor=:yellow,
    legend=false
)

plot!(colline2, 
    markersize=0,
    markerstrokestyle=:line,# :dot
    linewidth=1.25,
    linealpha=0.75,
    linecolor=:green,
    legend=false
)

# gt-box
plot!(gtboxplot,  
    linewidth=1.25,
    linestyle=:dot,
    linecolor=:red, 
    fillalpha=0, 
    legend=false
)
# dt-box
plot!(dtboxplot,  
    linewidth=1.25,
    linecolor=:blue, 
    fillalpha=0, 
    legend=false
)