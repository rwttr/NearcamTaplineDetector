using Flux
using Images
using ImageView
using Functors
using StatsBase
using Statistics
using CUDA

import LinearAlgebra
import Zygote

""" YOLO abstraction layers """
struct YoloLayer
    anchors                 # tuple of anchor boxes 
    net_inputsize           # network input size    
    num_cls                 # number of object class 
    featuremap_size         # featuremap size to perform detection [w h]
    featuremap_stride       # ratio of featuremap size to network input size    
    num_anchors             # number of anchors at a neuron
    x_grid_offset           # boxcenter x offseted by x_grid
    y_grid_offset           # boxcenter y offseted by y_grid
end

# constructor
function YoloLayer(anchors, net_inputsize, featuremap_size; num_cls=1)
    featuremap_stride = net_inputsize ./ featuremap_size;
    
    # rescale anchors' width and height to featuremap size
    anchors = collect(anchors) ./ featuremap_stride[1];
    num_anchors = length(anchors);

    x_coor = 1:featuremap_size[1];
    y_coor = 1:featuremap_size[2];

    y_grid = (ones(y_coor) * y_coor')'; # x-position grid coordinate
    x_grid = ones(x_coor) * x_coor';    # y-position grid coordinate
    x_grid_offset = x_grid ;            # mesh offset x
    y_grid_offset = y_grid ;            # mesh offset y

    YoloLayer(
        map(x -> Float32.(x), anchors),                # row vector [w h] rescaled to featuremap
        net_inputsize, 
        num_cls,                # number of class to be detected
        featuremap_size[1],     # squre featuremap
        Float32(featuremap_stride[1]),   # squre featuremap
        num_anchors,
        x_grid_offset,
        y_grid_offset
    );
end



""" Anchors Esimation """
##
# load dataset for anchor estimation
include("NearcamTaplineDataset.jl")
import .NearcamTaplineDataset as NDS
NDS.init();

# estimate anchors on testdata
ds1 = NDS.dataFold_1.test_data;
ds2 = NDS.dataFold_2.test_data;
ds3 = NDS.dataFold_3.test_data;
ds4 = NDS.dataFold_4.test_data;
ds5 = NDS.dataFold_5.test_data;

# anchors size estimate for nn_inputsize
nn_inputsize = [224 224];   # network input size 

anchors_k1 = NDS.estimateAnchors(ds1, nn_inputsize) |> collect
anchors_k2 = NDS.estimateAnchors(ds2, nn_inputsize) |> collect
anchors_k3 = NDS.estimateAnchors(ds3, nn_inputsize) |> collect
anchors_k4 = NDS.estimateAnchors(ds4, nn_inputsize) |> collect
anchors_k5 = NDS.estimateAnchors(ds5, nn_inputsize) |> collect

anchors = mean([anchors_k1 anchors_k2 anchors_k3 anchors_k4 anchors_k5], dims=2);
anchors = map(x -> Float32.(x), anchors); # Convert to Float32



""" # Model Building Section
    -load BackboneNet
    Pixel-level prediction with Box Detector
    YOLO + UNet (shared weights and computation)
"""
## load BackboneNet (network Architecture)
include("BackboneNet.jl");
import .BackboneNet as bb
basenet_darknetlight = bb.buildDarknetLight()

#= basenet = downsampling network
    #   tensor sizes
    1. (224, 224, 32, 1)
    2. (224, 224, 32, 1)

    3. (112, 112, 64, 1)
    4. (112, 112, 64, 1)
    5. (112, 112, 64, 1)

    6. (56, 56, 128, 1)
    7. (56, 56, 128, 1)
    8. (56, 56, 128, 1)

    9. (28, 28, 256, 1)
    10. (28, 28, 256, 1)
    11. (28, 28, 256, 1)

    12. (14, 14, 384, 1)
    13. (14, 14, 384, 1)
    14. (14, 14, 384, 1)

    15. (7, 7, 512, 1)
    16. (7, 7, 512, 1)
    17. (7, 7, 512, 1) 
=#

# load all model support functions
include("supportfunctions.jl")

""" Custom Layers """
# split layer def
struct Split{T}
    paths::T
end
Split(paths...) = Split(paths)  
@functor Split  
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths) # forward pass declaration

# upsampling layer def
struct UpBlock
    upsample
end
@functor UpBlock

function UpBlock(ind::Int, outd::Int) 
    UpBlock(
        Chain(
            ConvTranspose((2, 2), ind => outd, stride=(2, 2)),
		    BatchNorm(outd, leakyrelu)
        )
    )
end
function (u::UpBlock)(x, y)
    x = u.upsample(x)
    return cat(x, y, dims=3)
end



""" YOLO Branch .. Instance declaration """
# initialize yololayers
yololayer_layer1 = YoloLayer(anchors, nn_inputsize, [14 14]);
yololayer_layer2 = YoloLayer(anchors, nn_inputsize, [7 7]);

function yoloHeadBlock(inChannel, yololayer::YoloLayer)
    layers = [];
    num_anchors = yololayer.num_anchors;
    num_cls = yololayer.num_cls;

    push!(layers, Conv((1, 1), inChannel => num_anchors, tanh));    # tx for all anchors
    push!(layers, Conv((1, 1), inChannel => num_anchors, tanh));    # ty for all anchors
    push!(layers, Conv((1, 1), inChannel => num_anchors));          # tw for all anchors
    push!(layers, Conv((1, 1), inChannel => num_anchors));          # th for all anchors
    push!(layers, Conv((1, 1), inChannel => num_anchors, sigmoid)); # objectness
    push!(layers, Conv((1, 1), inChannel => num_anchors * num_cls, sigmoid)); # ty for all anchors

    return layers;
end



""" The Models """
struct NNModel
    basenet_block       # downsampling blocks
    upsample_block      # upsampling blocks
    yolohead_block      # box output
    pxpredhead_block    # pixel output
end
@functor NNModel

# Model_1 : network's forward pass
function (u::NNModel)(x::AbstractArray)
    # downsample
    x1 = u.basenet_block[1](x)
    x2 = u.basenet_block[2](x1)
    x3 = u.basenet_block[3](x2)
    x4 = u.basenet_block[4](x3)
    x5 = u.basenet_block[5](x4)
    x6 = u.basenet_block[6](x5)

    boxout1 = u.yolohead_block[1](x5)
    boxout2 = u.yolohead_block[2](x6)
    
    up_x5 = u.upsample_block[5](x6, x5)
    up_x4 = u.upsample_block[4](up_x5, x4)
    up_x3 = u.upsample_block[3](up_x4, x3)
    up_x2 = u.upsample_block[2](up_x3, x2)
    up_x1 = u.upsample_block[1](up_x2, x1)

    pxout = u.pxpredhead_block(up_x1)

    return boxout1, boxout2, pxout
end

""" Model_1_v1
    Variant_1 : column-wise softmax on pxl prediction output 
"""
function NNModel_v1(basenet, yololayer)
    basenet_block = (
        # basenet_1x 224 224 32
        basenet[1:2],
        # basenet_2x 112 112 64
        basenet[3:5],
        # basenet_4x 56 56 128
        basenet[6:8],
        # basenet_8x 28 28 256
        basenet[9:11],
        # basenet_16x 14 14 384
        basenet[12:14],
        # basenet_32x 7 7 512
        basenet[15:17]
    );
    
    up_32x16 = UpBlock(512, 128)        # from basenet_32x 7 7 512
    up_16x8 = UpBlock(384 + 128, 128)   # from basenet_16x 14 14 384
    up_8x4 = UpBlock(256 + 128, 64)     # from basenet_8x 28 28 256
    up_4x2 = UpBlock(128 + 64, 64)      # from basenet_4x 56 56 128
    up_2x1 = UpBlock(64 + 64, 32)       # from basenet_2x 112 112 64

    upsample_block = (
        up_2x1,
        up_4x2,
        up_8x4,
        up_16x8,
        up_32x16        
    )

    yoloBranch_1 = Chain(
        Conv((3, 3), 384 => 512, pad=SamePad()),
        BatchNorm(512, leakyrelu),
        Split(yoloHeadBlock(512, yololayer[1])...)  # output  
    )

    yoloBranch_2 = Chain(
        Conv((3, 3), 512 => 512, pad=SamePad()), 
        BatchNorm(512, leakyrelu),
        Split(yoloHeadBlock(512, yololayer[2])...)  # output
    )

    yolohead_block = yoloBranch_1, yoloBranch_2

    # tensor input coming from up_2x1
    pxpredhead_block = Chain(
        Conv((1, 1), 64 => 1, pad=SamePad(), tanh), # pointwise conv
        x -> softmax(x, dims=1) # column-wise softmax
    ) 

    NNModel(basenet_block, upsample_block, yolohead_block, pxpredhead_block)
end

model_1v1 = NNModel_v1(basenet_darknetlight, (yololayer_layer1, yololayer_layer2))


""" Model_1_v2
    Variant_2 : pixelwise sigmoid output 
"""
function NNModel_v2(basenet, yololayer)
    basenet_block = (
        # basenet_1x 224 224 32
        basenet[1:2],
        # basenet_2x 112 112 64
        basenet[3:5],
        # basenet_4x 56 56 128
        basenet[6:8],
        # basenet_8x 28 28 256
        basenet[9:11],
        # basenet_16x 14 14 384
        basenet[12:14],
        # basenet_32x 7 7 512
        basenet[15:17]
    );
    
    up_32x16 = UpBlock(512, 128)        # from basenet_32x 7 7 512
    up_16x8 = UpBlock(384 + 128, 128)   # from basenet_16x 14 14 384
    up_8x4 = UpBlock(256 + 128, 64)     # from basenet_8x 28 28 256
    up_4x2 = UpBlock(128 + 64, 64)      # from basenet_4x 56 56 128
    up_2x1 = UpBlock(64 + 64, 32)       # from basenet_2x 112 112 64

    upsample_block = (
        up_2x1,
        up_4x2,
        up_8x4,
        up_16x8,
        up_32x16        
    )

    yoloBranch_1 = Chain(
        Conv((3, 3), 384 => 512, pad=SamePad()),
        BatchNorm(512, leakyrelu),
        Split(yoloHeadBlock(512, yololayer[1])...)  # output  
    )

    yoloBranch_2 = Chain(
        Conv((3, 3), 512 => 512, pad=SamePad()), 
        BatchNorm(512, leakyrelu),
        Split(yoloHeadBlock(512, yololayer[2])...)  # output
    )

    yolohead_block = yoloBranch_1, yoloBranch_2

    # tensor input coming from up_2x1
    pxpredhead_block = Chain(
        Conv((1, 1), 64 => 1, pad=SamePad(), sigmoid), # pointwise conv
        # x -> softmax(x, dims=1) # column-wise softmax
    ) 

    NNModel(basenet_block, upsample_block, yolohead_block, pxpredhead_block)
end

model_1v2 = NNModel_v2(basenet_darknetlight, (yololayer_layer1, yololayer_layer2))

# Model_A
""" Model_A: parallel-supervised layers + col-wise softmax """

# Model_A
struct NNModelA
    basenet_block       # downsampling blocks
    upsample_block      # upsampling blocks
    yolohead_block      # box output
    pxpredhead_block    # pixel output
end
@functor NNModelA

# network's forward pass
function (u::NNModelA)(x::AbstractArray)
    # downsample
    x1 = u.basenet_block[1](x)
    x2 = u.basenet_block[2](x1)
    x3 = u.basenet_block[3](x2)
    x4 = u.basenet_block[4](x3)
    x5 = u.basenet_block[5](x4)
    x6 = u.basenet_block[6](x5)

    boxout1 = u.yolohead_block[1](x5)
    boxout2 = u.yolohead_block[2](x6)
    
    up_x5 = u.upsample_block[5](x6, x5)
    up_x4 = u.upsample_block[4](up_x5, x4)
    up_x3 = u.upsample_block[3](up_x4, x3)
    up_x2 = u.upsample_block[2](up_x3, x2)
    up_x1 = u.upsample_block[1](up_x2, x1)

    pxsup_1 = u.pxpredhead_block[1](up_x1)
    pxsup_2 = u.pxpredhead_block[2](up_x1)
    pxsup_3 = u.pxpredhead_block[3](up_x1)

    pxout = softmax(pxsup_1 + pxsup_2 + pxsup_3, dims=1)

    #return boxout1, boxout2, pxsup_1, pxsup_2, pxsup_3, pxout
    return boxout1, boxout2, pxout, pxsup_1, pxsup_2, pxsup_3
end

function NNModel_A1(basenet, yololayer)
    basenet_block = (
        # basenet_1x 224 224 32
        basenet[1:2],
        # basenet_2x 112 112 64
        basenet[3:5],
        # basenet_4x 56 56 128
        basenet[6:8],
        # basenet_8x 28 28 256
        basenet[9:11],
        # basenet_16x 14 14 384
        basenet[12:14],
        # basenet_32x 7 7 512
        basenet[15:17]
    );
    
    # upsample block
    up_32x16 = UpBlock(512, 128)        # from basenet_32x 7 7 512
    up_16x8 = UpBlock(384 + 128, 128)   # from basenet_16x 14 14 384
    up_8x4 = UpBlock(256 + 128, 64)     # from basenet_8x 28 28 256
    up_4x2 = UpBlock(128 + 64, 64)      # from basenet_4x 56 56 128
    up_2x1 = UpBlock(64 + 64, 32)       # from basenet_2x 112 112 64

    upsample_block = (
        up_2x1,
        up_4x2,
        up_8x4,
        up_16x8,
        up_32x16        
    )

    # yolo-block
    yoloBranch_1 = Chain(
        Conv((3, 3), 384 => 512, pad=SamePad()),
        BatchNorm(512, leakyrelu),
        Split(yoloHeadBlock(512, yololayer[1])...)  # output  
    )

    yoloBranch_2 = Chain(
        Conv((3, 3), 512 => 512, pad=SamePad()), 
        BatchNorm(512, leakyrelu),
        Split(yoloHeadBlock(512, yololayer[2])...)  # output
    )

    yolohead_block = yoloBranch_1, yoloBranch_2

    # pxl supervising block
    pxlsup_1 = Conv((1, 1), 64 => 1, pad=SamePad(), sigmoid)
    pxlsup_2 = Conv((1, 1), 64 => 1, pad=SamePad(), sigmoid)
    pxlsup_3 = Conv((1, 1), 64 => 1, pad=SamePad(), sigmoid)
    
    pxpredhead_block = pxlsup_1, pxlsup_2, pxlsup_3

    NNModelA(
        basenet_block, 
        upsample_block, 
        yolohead_block, 
        pxpredhead_block
    )
end

model_a = NNModel_A1(basenet_darknetlight, (yololayer_layer1, yololayer_layer2))



println("all model loaded");