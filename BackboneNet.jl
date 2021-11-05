""" 
This file contains raw cnn network based on Flux layers 

- ResNet-18, -34 (Raw)
- ResNet-50 (Pretrained) 
- MobileNetV2 (Raw)
- Darknet-53 (Raw)

"""
module BackboneNet

using Base:module_build_id
using Flux
using Flux:Data.DataLoader
using Functors
using Images
using ImageView
using CUDA
using BSON

export printchain
export buildresnet18
export buildresnet18_fcn
export buildresnet50_fcn

# print chained layers, spaitial and depth size at any layer
function printchain(fluxchain; inputTensor=[])
    
    if isempty(inputTensor)
        for i = 1:length(fluxchain)
            println(string(fluxchain[i]));
        end
    else        
        inputTensor =  Float32.(inputTensor);
        for i = 1:length(fluxchain)
            fluxchaintemp = fluxchain[1:i];
            outputTensor = fluxchaintemp(inputTensor);
            # println( string(fluxchain[i]))
            println("$i" * ". " * string(size(outputTensor)));
        end
    end

end

"""
Raw (unlearned weights) ResNet Flux Chains

Reference: 
cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
"""

"""
ResNet, Subblock type2 
Contain two conv-layers, used in ResNet18, ResNet34 Variants
"""

function generateSubBlockType2(;ind, outd, nfilter,fstride=1)
    # Subblock type2 = residual block with 2 convolutional layers
    # for resnet18, resnet34
    # nfilter : number of conv filter
    # ind : input tensor depth
    # outd : output tensor depth = nfilter (type2)
    # fstride : subblock first layuer stride size (downsample)
    chainblock = Flux.Chain(
        Conv((3, 3), ind => nfilter;pad=SamePad(), stride=fstride), # stride size controled top stack
        BatchNorm(nfilter, relu),
        Conv((3, 3), nfilter => outd;pad=SamePad(), stride=1),
        BatchNorm(outd, relu)        
    )

    shortcut = Flux.Chain(
        Conv((1, 1), ind => outd, stride=2, pad=SamePad()),
        BatchNorm(outd) # non-relu 
    );
    
    if fstride == 1
        return Flux.SkipConnection(chainblock, +);
    else
        return Flux.Parallel(+, chainblock, shortcut);
    end
    
end

# standard resnet18
function buildresnet18(nclass;img_inputChannel=3)
    # Standard ResNet18 

    # Define resnet18 layers
    # --downsample 1
    conv1x = Conv((7, 7), img_inputChannel => 64, stride=2, pad=SamePad());   
    # --downsample 2
    maxpool_1 = MaxPool((3, 3), stride=2, pad=SamePad());
    subblock_conv2x_1 = generateSubBlockType2(ind=64, outd=64, nfilter=64, fstride=1);
    subblock_conv2x_2 = generateSubBlockType2(ind=64, outd=64, nfilter=64, fstride=1);
    # --downsample 3
    subblock_conv3x_1 = generateSubBlockType2(ind=64, outd=128, nfilter=128, fstride=2); 
    subblock_conv3x_2 = generateSubBlockType2(ind=128, outd=128, nfilter=128, fstride=1);
    # --downsample 4
    subblock_conv4x_1 = generateSubBlockType2(ind=128, outd=256, nfilter=256, fstride=2);
    subblock_conv4x_2 = generateSubBlockType2(ind=256, outd=256, nfilter=256, fstride=1); 
    # --downsample 5
    subblock_conv5x_1 = generateSubBlockType2(ind=256, outd=512, nfilter=512, fstride=2); 
    subblock_conv5x_2 = generateSubBlockType2(ind=512, outd=512, nfilter=512, fstride=1);

    avgpool_1 = MeanPool((7, 7), pad=0, stride=1);
    flat_1 = Flux.flatten;
    fc = Dense(512, nclass);
    
    # Assemble
    resnet18 = [];
    push!(resnet18, conv1x);
    push!(resnet18, maxpool_1);
    push!(resnet18, subblock_conv2x_1);
    push!(resnet18, subblock_conv2x_2);
    push!(resnet18, subblock_conv3x_1);
    push!(resnet18, subblock_conv3x_2);
    push!(resnet18, subblock_conv4x_1);
    push!(resnet18, subblock_conv4x_2);
    push!(resnet18, subblock_conv5x_1);
    push!(resnet18, subblock_conv5x_2);
    push!(resnet18, avgpool_1);
    push!(resnet18, flat_1);
    push!(resnet18, fc);
    push!(resnet18, softmax);

    return Chain(resnet18...);
end

# trimed resnet18 
function buildresnet18_fcn(;img_inputChannel=3)
    # standard resnet18 network
    basechain = buildresnet18(1;img_inputChannel=img_inputChannel);

    # remove 4 last layers, preserving convolutional layers
    # last layer produce : 7×7×512×1 Array{Float32, 4}
    basechain = basechain[1:end - 4];
    baselayers = collect(basechain); # for push! 

    return Chain(baselayers...);

end

# standard resnet34
function buildresnet34(nclass;img_inputChannel=3)
    # Standard ResNet34 

    # layers 
    # --downsample 1
    conv1x = Conv((7, 7), img_inputChannel => 64, stride=2, pad=SamePad());   
    # --downsample 2
    maxpool_1 = MaxPool((3, 3), stride=2, pad=SamePad());
    conv2x_1 = generateSubBlockType2(ind=64, outd=64, nfilter=64, fstride=1);
    conv2x_2 = generateSubBlockType2(ind=64, outd=64, nfilter=64, fstride=1);
    conv2x_3 = generateSubBlockType2(ind=64, outd=64, nfilter=64, fstride=1);
    # --downsample 3
    conv3x_1 = generateSubBlockType2(ind=64, outd=128, nfilter=128, fstride=2); 
    conv3x_2 = generateSubBlockType2(ind=128, outd=128, nfilter=128, fstride=1);
    conv3x_3 = generateSubBlockType2(ind=128, outd=128, nfilter=128, fstride=1);
    conv3x_4 = generateSubBlockType2(ind=128, outd=128, nfilter=128, fstride=1);
    # --downsample 4
    conv4x_1 = generateSubBlockType2(ind=128, outd=256, nfilter=256, fstride=2);
    conv4x_2 = generateSubBlockType2(ind=256, outd=256, nfilter=256, fstride=1);
    conv4x_3 = generateSubBlockType2(ind=256, outd=256, nfilter=256, fstride=1);
    conv4x_4 = generateSubBlockType2(ind=256, outd=256, nfilter=256, fstride=1);
    conv4x_5 = generateSubBlockType2(ind=256, outd=256, nfilter=256, fstride=1);
    conv4x_6 = generateSubBlockType2(ind=256, outd=256, nfilter=256, fstride=1); 
    # --downsample 5
    conv5x_1 = generateSubBlockType2(ind=256, outd=512, nfilter=512, fstride=2); 
    conv5x_2 = generateSubBlockType2(ind=512, outd=512, nfilter=512, fstride=1);
    conv5x_3 = generateSubBlockType2(ind=512, outd=512, nfilter=512, fstride=1);
    avgpool_1 = MeanPool((7, 7), pad=0, stride=1);
    flat_1 = Flux.flatten;
    fc = Dense(512, nclass);

    resnet34 = [];
    push!(resnet34, conv1x);
    push!(resnet34, maxpool_1, conv2x_1, conv2x_2, conv2x_3);
    push!(resnet34, conv3x_1, conv3x_2, conv3x_3, conv3x_4);
    push!(resnet34, conv4x_1, conv4x_2, conv4x_3, conv4x_4, conv4x_5, conv4x_6);
    push!(resnet34, conv5x_1, conv5x_2, conv5x_3);
    push!(resnet34, avgpool_1, flat_1, fc, softmax);

    return resnet34;

end

# trimed resnet34
function buildresnet34_fcn(;img_inputChannel=3)
    # standard resnet34 network
    basechain = buildresnet34(1;img_inputChannel=img_inputChannel);

    # remove 4 last layers, preserving convolutional layers
    # last layer produce : 7×7×512×1 Array{Float32, 4}
    basechain = basechain[1:end - 4];
    baselayers = collect(basechain); # for push! 

    return Chain(baselayers...);    
end

# ResNet50 with weight from Metalhead.jl
struct ResidualBlock
    conv_layers
    norm_layers
    shortcut
end

@functor ResidualBlock

function ResidualBlock(filters, kernels::Array{Tuple{Int,Int}}, pads::Array{Tuple{Int,Int}}, 
    strides::Array{Tuple{Int,Int}}, shortcut=identity)
    local conv_layers = []
    local norm_layers = []
    for i in 2:length(filters)
        push!(conv_layers, Conv(kernels[i - 1], filters[i - 1] => filters[i], pad=pads[i - 1], stride=strides[i - 1]))
        push!(norm_layers, BatchNorm(filters[i]))
    end
    ResidualBlock(Tuple(conv_layers), Tuple(norm_layers), shortcut)
end

function ResidualBlock(filters, kernels::Array{Int}, pads::Array{Int}, strides::Array{Int}, shortcut=identity)
    ResidualBlock(filters, [(i, i) for i in kernels], [(i, i) for i in pads], [(i, i) for i in strides], shortcut)
end

function (block::ResidualBlock)(input)
    local value = copy.(input)
    for i in 1:length(block.conv_layers) - 1
        value = relu.((block.norm_layers[i])((block.conv_layers[i])(value)))
    end
    relu.(((block.norm_layers[end])((block.conv_layers[end])(value))) + block.shortcut(input))
end

function Bottleneck(filters::Int, downsample::Bool=false, res_top::Bool=false)
    if (!downsample && !res_top)
        return ResidualBlock([4 * filters, filters, filters, 4 * filters], [1,3,1], [0,1,0], [1,1,1])
    elseif (downsample && res_top)
        return ResidualBlock(
            [filters, filters, filters, 4 * filters], [1,3,1], [0,1,0], [1,1,1], 
            Chain(Conv((1, 1), filters => 4 * filters, pad=(0, 0), stride=(1, 1)), BatchNorm(4 * filters)))
    else
        shortcut = Chain(Conv((1, 1), 2 * filters => 4 * filters, pad=(0, 0), stride=(2, 2)), BatchNorm(4 * filters))
        return ResidualBlock([2 * filters, filters, filters, 4 * filters], [1,3,1], [0,1,0], [1,1,2], shortcut)
    end
end

# Resnet-50 raw layers
function resnet50()
    local layers = [3, 4, 6, 3]
    local layer_arr = []

    push!(layer_arr, Conv((7, 7), 3 => 64, pad=(3, 3), stride=(2, 2)))
    push!(layer_arr, MaxPool((3, 3), pad=(1, 1), stride=(2, 2)))

    initial_filters = 64
    for i in 1:length(layers)
        push!(layer_arr, Bottleneck(initial_filters, true, i == 1))
        for j in 2:layers[i]
            push!(layer_arr, Bottleneck(initial_filters))
        end
        initial_filters *= 2
    end

    push!(layer_arr, MeanPool((7, 7)))
    push!(layer_arr, x -> reshape(x, :, size(x, 4)))
    push!(layer_arr, (Dense(2048, 1000)))
    push!(layer_arr, softmax)

    Chain(layer_arr...)
end

# load weight into resnet50
function resnet50_weights()
    weight = BSON.load("weights/resnet.bson")
    weights = Dict{Any,Any}()
    for ele in keys(weight)
        weights[string(ele)] = convert(Array{Float64,N} where N, weight[ele])
    end
    ls = resnet50()
    ls[1].weight .= weights["gpu_0/conv1_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:]
    count = 2
    for j in [3:5, 6:9, 10:15, 16:18]
        for p in j
            ls[p].conv_layers[1].weight .= weights["gpu_0/res$(count)_$(p - j[1])_branch2a_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:]
            ls[p].conv_layers[2].weight .= weights["gpu_0/res$(count)_$(p - j[1])_branch2b_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:]
            ls[p].conv_layers[3].weight .= weights["gpu_0/res$(count)_$(p - j[1])_branch2c_w_0"][end:-1:1,:,:,:][:,end:-1:1,:,:]
        end
        count += 1
    end
    ls[21].W .= transpose(weights["gpu_0/pred_w_0"]); ls[21].b .= weights["gpu_0/pred_b_0"]
    return ls
end

function buildresnet50_fcn()
    basechain = resnet50_weights();
    # remove 4 last layers, preserving convolutional layers
    # last layer produce : 7×7×512×1 Array{Float32, 4}
    basechain = basechain[1:end - 4];    
    return basechain
end

"""
raw MobileNetV2 

This implementation follows 
MobileNetV2: Inverted Residuals and Linear Bottlenecks, https://arxiv.org/pdf/1801.04381v4.pdf

Note that, the paper stated that MobileNetV2 has 19 bottleneck residual blocks
but in table-1 only show the configuration in 17 blocks 
"""
# moded swish * relu6 activation function
function swish6(x)
    return min(max(0, Flux.swish(x)), 6)
end

function swish6(x::AbstractArray, args...)
    return swish6(x)
end

function generatebtlneck(;ind=1,outd=1,bstride=1,expfactor=1, f=false)
    # generate mobilenet bottleneck unit
    # bstride: block stride (s)
    # ind: block input depth
    # outd: block output depth
    # expfactor: expansion factor (t)
    # f: block first layer indicator

    bottleneckresblock = Chain(
        Conv((1, 1), ind => (expfactor * ind), pad=SamePad()),
        BatchNorm(expfactor * ind, relu6),
        DepthwiseConv((3, 3), (expfactor * ind) => (expfactor * ind), stride=bstride, pad=SamePad()),
        BatchNorm(expfactor * ind, relu6),
        Conv((1, 1), (expfactor * ind) => outd, pad=SamePad()),
        BatchNorm(outd)
    );

    if (bstride == 1) && (f == true)
        return bottleneckresblock;
    elseif bstride == 1        
        return Flux.SkipConnection(bottleneckresblock, +);
    else
        return bottleneckresblock;
    end
end

function buildMobilenetV2(;num_cls=1, img_inputchannel=3)

    # MobileNetV2 block structure
    mobilenetlayers = [];
    push!(mobilenetlayers,
        # 224 x 224 x 3
        "conv2d_1" => Conv((3, 3), img_inputchannel => 32, stride=2, pad=SamePad()),
        # 112 x 112 x 32
        "block1_1" => generatebtlneck(ind=32, outd=16, bstride=1, expfactor=1, f=true),
        # 112 x 112 x 16
        "block2_1" => generatebtlneck(ind=16, outd=24, bstride=2, expfactor=6),
        "block2_2" => generatebtlneck(ind=24, outd=24, bstride=1, expfactor=6),
        # 56 x 56 x 24
        "block3_1" => generatebtlneck(ind=24, outd=32, bstride=2, expfactor=6),
        "block3_2" => generatebtlneck(ind=32, outd=32, bstride=1, expfactor=6),
        "block3_3" => generatebtlneck(ind=32, outd=32, bstride=1, expfactor=6),
        # 28 x 28 x 32
        "block4_1" => generatebtlneck(ind=32, outd=64, bstride=2, expfactor=6),
        "block4_2" => generatebtlneck(ind=64, outd=64, bstride=1, expfactor=6),
        "block4_3" => generatebtlneck(ind=64, outd=64, bstride=1, expfactor=6),
        "block4_4" => generatebtlneck(ind=64, outd=64, bstride=1, expfactor=6),
        # 14 x 14 x 64
        "block5_1" => generatebtlneck(ind=64, outd=96, bstride=1, expfactor=6, f=true),
        "block5_2" => generatebtlneck(ind=96, outd=96, bstride=1, expfactor=6),
        "block5_3" => generatebtlneck(ind=96, outd=96, bstride=1, expfactor=6),
        # 14 x 14 x 96
        "block6_1" => generatebtlneck(ind=96, outd=160, bstride=2, expfactor=6),
        "block6_2" => generatebtlneck(ind=160, outd=160, bstride=1, expfactor=6),
        "block6_3" => generatebtlneck(ind=160, outd=160, bstride=1, expfactor=6),
        # 7 x 7 x 160
        "block7_1" => generatebtlneck(ind=160, outd=320, bstride=1, expfactor=6, f=true),
        # 7 x 7 x 320 
        "conv2d_2" => Conv((1, 1), 320 => 1280, pad=SamePad()),
        # 7 x 7 x 1280
        "avgpool7" => MeanPool((7, 7)),
        # 1 x 1 x 1280
        "conv2d_3" => Conv((1, 1), 1280 => num_cls)
    );

    chainnet = Chain(map(x -> x.second, mobilenetlayers)...)
    return mobilenetlayers, chainnet;
end

"""
Raw Darknet-53 
implementation follows YOLOv3 paper https://arxiv.org/pdf/1804.02767v1.pdf

"""

function darknetResBlock(;ind, selfrep=1)
    # ind = input depth
    # fnfilter = no of filter at fisrt conv layer
    # selfrep = self repeat count    
    fnfilter = Int(ind / 2);
    # Residual Block 
    blocklayers =  Flux.SkipConnection(                                        
        Chain(                            
            Conv((1, 1), ind => fnfilter, pad=SamePad()),
            BatchNorm(fnfilter, leakyrelu),
            Conv((3, 3), fnfilter => fnfilter * 2, pad=SamePad()),
            BatchNorm(fnfilter * 2, leakyrelu)
        ),+
    )
    
    stacklayer = [];
    for i = 1:selfrep
        push!(stacklayer, blocklayers)
    end

    return Chain(stacklayer...);
end

function buildDarknet53(;input_channel=3)

    darknetlayers = Chain(
        Conv((3, 3), input_channel => 32, pad=SamePad()),
        Conv((3, 3), 32 => 64, pad=SamePad(), stride=2),
        darknetResBlock(ind=64, selfrep=1)...,
        Conv((3, 3), 64 => 128, pad=SamePad(), stride=2),
        darknetResBlock(ind=128, selfrep=2)...,
        Conv((3, 3), 128 => 256, pad=SamePad(), stride=2),
        darknetResBlock(ind=256, selfrep=8)...,
        Conv((3, 3), 256 => 512, pad=SamePad(), stride=2),
        darknetResBlock(ind=512, selfrep=8)...,
        Conv((3, 3), 512 => 1024, pad=SamePad(), stride=2),
        darknetResBlock(ind=1024, selfrep=4)...,
        GlobalMeanPool(),
        flatten,
        Dense(1024, 1000),
        softmax
    )

    return darknetlayers
end

function buildDarknet53_fcn(;input_channel=3)
    darknetlayers = buildDarknet53(input_channel=input_channel) |> collect
    return Chain(darknetlayers[1:end - 4]...)
end


"""
Darknet-light

- reduced version of darknet-53

"""
# fully conv network
function buildDarknetLight(;input_channel=3)

    darknetlayers = Chain(
        Conv((3, 3), input_channel => 32, pad=SamePad()),
        BatchNorm(32, leakyrelu),
        
        Conv((3, 3), 32 => 64, pad=SamePad(), stride=2),
        BatchNorm(64, leakyrelu),
        
        darknetResBlock(ind=64, selfrep=1),
        
        Conv((3, 3), 64 => 128, pad=SamePad(), stride=2),
        BatchNorm(128, leakyrelu),
        
        darknetResBlock(ind=128, selfrep=2),
        
        Conv((3, 3), 128 => 256, pad=SamePad(), stride=2),
        BatchNorm(256, leakyrelu),
        
        darknetResBlock(ind=256, selfrep=4),

        Conv((3, 3), 256 => 384, pad=SamePad(), stride=2),
        BatchNorm(384, leakyrelu),

        darknetResBlock(ind=384, selfrep=4),

        Conv((3, 3), 384 => 512, pad=SamePad(), stride=2),
        BatchNorm(512, leakyrelu),

        darknetResBlock(ind=512, selfrep=1)
    )

    return darknetlayers
end

function buildDarknetLight2(;input_channel=3)
    
    darknetlayers = Chain(
        Conv((3, 3), input_channel => 32, pad=SamePad()),
        BatchNorm(32, leakyrelu),
        
        Conv((3, 3), 32 => 64, pad=SamePad(), stride=2),
        BatchNorm(64, leakyrelu),
        
        darknetResBlock(ind=64, selfrep=1),
        
        Conv((3, 3), 64 => 128, pad=SamePad(), stride=2),
        BatchNorm(128, leakyrelu),
        
        darknetResBlock(ind=128, selfrep=1),
        
        Conv((3, 3), 128 => 256, pad=SamePad(), stride=2),
        BatchNorm(256, leakyrelu),
        
        darknetResBlock(ind=256, selfrep=2),

        Conv((3, 3), 256 => 320, pad=SamePad(), stride=2),
        BatchNorm(320, leakyrelu),

        darknetResBlock(ind=320, selfrep=2),

        Conv((3, 3), 320 => 384, pad=SamePad(), stride=2),
        BatchNorm(384, leakyrelu),

        darknetResBlock(ind=384, selfrep=1)
    )

    return darknetlayers

end

end #module