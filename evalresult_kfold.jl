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

"""  Bounding Box Eval """
# eval function for single bbox iou
function evalSingleIOU(result_model_path::String; iou_eval=0.5)

    ap_box_fold = zeros(Float32, 5)             # average precision for each fold
    dh_fold_vector = Vector(undef, 5)           # vector of dh for each fold
    eperr_fold_vector = Vector(undef, 5)        # vector of endpoint error for each fold
    mask_fold_vector = Vector(undef, 5)         # positive detection mask

    dh_avg_fold = zeros(Float32, 5)             # avg dh of positive detection
    eperr_avg_fold = zeros(Float32, 5)          # avg endpoint error of positive detection

    Threads.@threads for fold_no = 1:5
        local dtboxiou
        local dttaplineimg
        local gttapline

        @load joinpath(result_model_path, "dtboxiou_fold$fold_no.bson") dtboxiou
        @load joinpath(result_model_path,"dttapline_fold$fold_no.bson") dttaplineimg
        @load "./data/nearcam_fold_gt/gttapline_fold$fold_no.bson" gttapline

        ap_box_fold[fold_no] = count(x -> x >= iou_eval, dtboxiou) / length(dtboxiou)
        dh_fold_vector[fold_no] = taplineHausdorffDist(gttapline, dttaplineimg)
        eperr_fold_vector[fold_no] = endpointErr(gttapline, dttaplineimg)

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
    end

    ap_model = Statistics.mean(ap_box_fold);
    avg_detected_dh = Statistics.mean(dh_avg_fold)
    avg_detected_eperr = Statistics.mean(eperr_avg_fold)

    @show result_model_path;
    @show iou_eval;
    @show ap_box_fold;                  # ap by fold
    @show ap_model;                     # overall model ap
    @show avg_detected_dh;              # average dh of positive-detected of overall model
    @show avg_detected_eperr            # average endpoint error of positive-detected of overall model
    @show dh_avg_fold;                  # average dh of positive-detected by fold
    @show eperr_avg_fold                # average endpoint error of positive-detected by fold
    @show avg_detected_dh / (224 * sqrt(2)) * 100; # error percent (relative to image size)
    @show Statistics.mean.(dh_fold_vector);

    avg_model_dh = Statistics.mean(Statistics.mean.(dh_fold_vector));
    @show avg_model_dh;

    avg_model_eperr = Statistics.mean(Statistics.mean.(eperr_fold_vector));
    @show avg_model_eperr;
end


##
res_dir= readdir("./result", join=true)
for i in res_dir
    evalSingleIOU(i)
    println("")
end