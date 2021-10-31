using BSON:@load
using BSON:@save
using Images
using ImageView
import Statistics
import LinearAlgebra

# hausdorff distance between 2 polyline images
function taplineHausdorffDist(imgA::AbstractArray{Bool,2}, imgB::AbstractArray{Bool,2})
    pt_a = findall(imgA);
    pt_b = findall(imgB);

    if isempty(pt_a) || isempty(pt_b)
        return -1
    end

    dh = Images.hausdorff(imgA, imgB)
    return dh
end
function taplineHausdorffDist(imgA::AbstractArray{Bool,3}, imgB::AbstractArray{Bool,3})
    dh_array = taplineHausdorffDist.(eachslice(imgA, dims=3), eachslice(imgB, dims=3))
    return dh_array
end

# Tapline endpoint error distance
function endpointErr(imgA::AbstractArray{Bool,2}, imgB::AbstractArray{Bool,2})
    pt_a = findall(imgA);
    pt_b = findall(imgB);

    if isempty(pt_a) || isempty(pt_b)
        return -1
    end

    err_1 = LinearAlgebra.norm((pt_a[1] - pt_b[1]).I) 
    err_2 = LinearAlgebra.norm((pt_a[end] - pt_b[end]).I) 

    return err_1 + err_2
end
function endpointErr(imgA::AbstractArray{Bool,3}, imgB::AbstractArray{Bool,3})
    err_array = endpointErr.(eachslice(imgA, dims=3), eachslice(imgB, dims=3))
    return err_array
end

function dicef1score(imgA::AbstractArray{Bool,2}, imgB::AbstractArray{Bool,2})
    intersected_px = count(x -> x == 1, imgA .* imgB);
    total_px = count(x -> x == 1, imgA) + count(x -> x == 1, imgB)
    f1score = (2 * intersected_px) / total_px
    return f1score
end
function dicef1score(imgA::AbstractArray{Bool,3}, imgB::AbstractArray{Bool,3})
    f1score = dicef1score.(eachslice(imgA, dims=3), eachslice(imgB, dims=3))
    return f1score
end

"""  Bounding Box Eval """
# eval function for single bbox iou
function evalSingleIOU(result_model_path::String; iou_eval=0.5, kfold=5)

    ap_box_fold = zeros(Float32, kfold)         # average precision for each fold
    dh_fold_vector = Vector(undef, kfold)       # vector of dh for each fold
    eperr_fold_vector = Vector(undef, kfold)    # vector of endpoint error for each fold    
    dicef1_fold_vector = Vector(undef, kfold)   # vector of f1 score for each fold
    
    # box desire
    mask_fold_vector = Vector(undef, kfold)     # positive detection mask
    dh_avg_fold = zeros(Float32, kfold)         # fold-avg dh of positive detection
    eperr_avg_fold = zeros(Float32, kfold)      # fold-avg endpoint error of positive detection
    f1_avg_fold = zeros(Float32, kfold)          # fold-avg f1-score of positive detection

    Threads.@threads for fold_no = 1:kfold
        local dtboxiou
        local dttaplineimg
        local gttapline

        @load joinpath(result_model_path, "dtboxiou_fold$fold_no.bson") dtboxiou
        @load joinpath(result_model_path, "dttapline_fold$fold_no.bson") dttaplineimg
        @load "./data/nearcam_fold_gt/gttapline_fold$fold_no.bson" gttapline

        ap_box_fold[fold_no] = count(x -> x >= iou_eval, dtboxiou) / length(dtboxiou)
        dh_fold_vector[fold_no] = taplineHausdorffDist(gttapline, dttaplineimg)
        eperr_fold_vector[fold_no] = endpointErr(gttapline, dttaplineimg)
        dicef1_fold_vector[fold_no] = dicef1score(gttapline, dttaplineimg)

        # mask invalids with NaN
        mask_fold_vector[fold_no] =  map(dtboxiou) do x
            if x >= iou_eval
                return 1
            else
                return NaN
            end
        end

        dh_masked = mask_fold_vector[fold_no] .* dh_fold_vector[fold_no]
        dh_avg_fold[fold_no] = Statistics.mean(filter(!isnan, dh_masked))

        eperr_masked = mask_fold_vector[fold_no] .* eperr_fold_vector[fold_no]
        eperr_avg_fold[fold_no] = Statistics.mean(filter(!isnan, eperr_masked))

        f1_masked = mask_fold_vector[fold_no] .* dicef1_fold_vector[fold_no]
        f1_avg_fold[fold_no] = Statistics.mean(filter(!isnan, f1_masked))
    end

    ap_model = Statistics.mean(ap_box_fold);

    avg_detected_dh = Statistics.mean(dh_avg_fold)
    avg_detected_dh_percent = avg_detected_dh / (224 * sqrt(2)) * 100;

    avg_detected_eperr = Statistics.mean(eperr_avg_fold)
    avg_detected_eperr_percent = avg_detected_eperr / (224 * sqrt(2)) * 100;

    avg_detected_f1score = Statistics.mean(f1_avg_fold)

    @show result_model_path;
    println("Box desire results")
    @show ap_box_fold;              # ap by fold
    @show ap_model;                 # overall model ap

    @show dh_avg_fold               # fold-avg dh of positive-detected
    @show avg_detected_dh;          # model-avg dh of positive-detected
    @show avg_detected_dh_percent;  # error percent (relative to image size)
        
    @show eperr_avg_fold
    @show avg_detected_eperr        # average endpoint error of positive-detected of overall model
    @show avg_detected_eperr_percent # error percent (relative to image size)

    
    @show avg_detected_f1score

    println("Standard Metrics")
    # @show Statistics.mean.(dh_fold_vector);                  # average dh of positive-detected by fold
    # @show Statistics.mean.(dicef1_fold_vector);
    # @show Statistics.mean.(eperr_fold_vector);                # average endpoint error of positive-detected by fold
    
    model_avg_dh = Statistics.mean(Statistics.mean.(dh_fold_vector))
    modeL_avg_eperr = Statistics.mean(Statistics.mean.(eperr_fold_vector))
    model_avg_f1 = Statistics.mean(Statistics.mean.(dicef1_fold_vector))

    @show model_avg_dh
    @show modeL_avg_eperr
    @show model_avg_f1
end


## eval all model in ./result
res_dir = readdir("./result", join=true)
for i in res_dir
    evalSingleIOU(i, iou_eval=0.75)
    println("")
end

##
evalSingleIOU("./result\\model_a", iou_eval=0.75)