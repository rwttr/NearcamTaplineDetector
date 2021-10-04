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

_testdata = [
    NDS.dataFold_1.test_data    
    NDS.dataFold_2.test_data
    NDS.dataFold_3.test_data
    NDS.dataFold_4.test_data
    NDS.dataFold_5.test_data
];

function detect_kfold(load_model_path::String, load_model_name::String, save_result_path::String, testdata_k; gpu_enable=true)   
    for fold_no = 1:5
        # init testdata
        testdata = testdata_k[fold_no]       
        NDS.resetDispatchRecord()
        current_data_iter = NDS.getDispatchRecord()
        max_data_iter = testdata.n

        # Load Model
        model_url = joinpath(load_model_path, "fold$fold_no", load_model_name) * "_fold$fold_no" * "_epoch100.bson"
        @load model_url model_save
        local model = model_save;
        if gpu_enable
            model = gpu(model);
        end

        # init results
        dtboxiou_array = zeros(testdata.n)
        dttapline_array = zeros(Bool, nn_inputsize..., testdata.n)
        gttapline_array = zeros(Bool, nn_inputsize..., testdata.n)

        counter_i = 1        
        while current_data_iter <= (max_data_iter - data_dispatch_size + 1)
            # dataloader
            testDL = NDS.dispatchData(testdata; dispatch_size=data_dispatch_size, shuffle_enable=false, img_outputsize=nn_inputsize);

            # update datastore index
            current_data_iter = NDS.getDispatchRecord()  

            # image data
            img = Float32.(testDL.data[1]) |> normalizeImg;
            if gpu_enable
                img = gpu(img);
            end
            # groundtruth
            gtbox = Float32.(testDL.data[2]);    
            gtpxl = Float32.(testDL.data[3]);
    
            # model inference
            # nnoutput = model(img)
            nnoutput = Base.invokelatest(model, img) # avoid definition from model.jl
            
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
        @save joinpath(save_result_path, "dtboxiou_fold$fold_no.bson") dtboxiou
        @save joinpath(save_result_path, "dttapline_fold$fold_no.bson") dttaplineimg
    end
    if gpu_enable
        CUDA.reclaim()
    end
end


## run detection

# 0.5 bbox iou penalty

# model_1v1_std_std_dice
load_model_path = "./weights/05_iou_penalty/model_1v1_std_std_dice"
load_model_name  = "model_1v1_std_std_dice"
save_result_path = "result/p5_iou_model_1v1_dice"
detect_kfold(load_model_path, load_model_name, save_result_path, _testdata)   

# model_1v1_std_std_focal
load_model_path = "./weights/05_iou_penalty/model_1v1_std_std_focal"
load_model_name  = "model_1v1_std_std_focal"
save_result_path = "result/p5_iou_model_1v1_focal"
detect_kfold(load_model_path, load_model_name, save_result_path, _testdata)  

# model_1v1_std_std_tversky
load_model_path = "./weights/05_iou_penalty/model_1v1_std_std_tversky"
load_model_name  = "model_1v1_std_std_tversky"
save_result_path = "result/p5_iou_model_1v1_tversky"
detect_kfold(load_model_path, load_model_name, save_result_path, _testdata)

# model_1v2_std_std_dice
load_model_path = "./weights/05_iou_penalty/model_1v2_std_std_dice"
load_model_name  = "model_1v2_std_std_dice"
save_result_path = "result/p5_iou_model_1v2_dice"
detect_kfold(load_model_path, load_model_name, save_result_path, _testdata)   

# model_1v2_std_std_focal
load_model_path = "./weights/05_iou_penalty/model_1v2_std_std_focal"
load_model_name  = "model_1v2_std_std_focal"
save_result_path = "result/p5_iou_model_1v2_focal"
detect_kfold(load_model_path, load_model_name, save_result_path, _testdata)  

# model_1v2_std_std_tversky
load_model_path = "./weights/05_iou_penalty/model_1v2_std_std_tversky"
load_model_name  = "model_1v2_std_std_tversky"
save_result_path = "result/p5_iou_model_1v2_tversky"
detect_kfold(load_model_path, load_model_name, save_result_path, _testdata)

# model_a
load_model_path = "./weights/05_iou_penalty/model_a"
load_model_name  = "model_a"
save_result_path = "result/p5_iou_model_a"
detect_kfold(load_model_path, load_model_name, save_result_path, _testdata)


# no bbox iou penalty

# model_1v1_std_std_dice
load_model_path = "./weights/no_iou_penalty/model_1v1_std_std_dice"
load_model_name  = "model_1v1_std_std_dice"
save_result_path = "result/no_iou_model_1v1_dice"
detect_kfold(load_model_path, load_model_name, save_result_path, _testdata)   

# model_1v1_std_std_focal
load_model_path = "./weights/no_iou_penalty/model_1v1_std_std_focal"
load_model_name  = "model_1v1_std_std_focal"
save_result_path = "result/no_iou_model_1v1_focal"
detect_kfold(load_model_path, load_model_name, save_result_path, _testdata)  

# model_1v1_std_std_tversky
load_model_path = "./weights/no_iou_penalty/model_1v1_std_std_tversky"
load_model_name  = "model_1v1_std_std_tversky"
save_result_path = "result/no_iou_model_1v1_tversky"
detect_kfold(load_model_path, load_model_name, save_result_path, _testdata)

# model_1v2_std_std_dice
load_model_path = "./weights/no_iou_penalty/model_1v2_std_std_dice"
load_model_name  = "model_1v2_std_std_dice"
save_result_path = "result/no_iou_model_1v2_dice"
detect_kfold(load_model_path, load_model_name, save_result_path, _testdata)   

# model_1v2_std_std_focal
load_model_path = "./weights/no_iou_penalty/model_1v2_std_std_focal"
load_model_name  = "model_1v2_std_std_focal"
save_result_path = "result/no_iou_model_1v2_focal"
detect_kfold(load_model_path, load_model_name, save_result_path, _testdata)  

# model_1v2_std_std_tversky
load_model_path = "./weights/no_iou_penalty/model_1v2_std_std_tversky"
load_model_name  = "model_1v2_std_std_tversky"
save_result_path = "result/no_iou_model_1v2_tversky"
detect_kfold(load_model_path, load_model_name, save_result_path, _testdata)

# model_a
load_model_path = "./weights/no_iou_penalty/model_a"
load_model_name  = "model_a"
save_result_path = "result/no_iou_model_a"
detect_kfold(load_model_path, load_model_name, save_result_path, _testdata)