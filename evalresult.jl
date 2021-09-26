using BSON:@load
using BSON:@save
using ImageView
import Statistics
import LinearAlgebra

# hausdorff distance between 2 polyline images
function taplineHausdorffDist(imgA::Array{T,2}, imgB::Array{T,2}) where T <: Real
    # hausdorf distance
    # drop singleton dimensions
    imgA_d = dropdims(imgA, dims=(findall(size(imgA) .== 1)...,))
    imgB_d = dropdims(imgB, dims=(findall(size(imgB) .== 1)...,))

    px_a = findall(Bool.(imgA_d));
    px_b = findall(Bool.(imgB_d));

    if isempty(px_a) || isempty(px_b)
        return -1;
    end

    # distance from a to b
    a_b_shortest = [];
    for a in px_a
        tempdist = [];
        for b in px_b
            push!(tempdist, LinearAlgebra.norm((a - b).I))
        end
        push!(a_b_shortest, findmin(tempdist)[1]);
    end

    # shortest distance from every point on b to a
    b_a_shortest = [];
    for b in px_b
        tempdist = [];
        for a in px_a
            push!(tempdist, LinearAlgebra.norm((a - b).I))
        end
        push!(b_a_shortest, findmin(tempdist)[1]);
    end

    # supremum
    sup_a_b = max(a_b_shortest...)
    sup_b_a = max(b_a_shortest...)

    return max(sup_a_b, sup_b_a);
end
# hausdorff distance between 2 polyline images in dims=3 pair
function taplineHausdorffDist(imgA::Array{T,3}, imgB::Array{T,3}) where T <: Real
    @assert size(imgA) == size(imgB)
    img_channel = size(imgA)[3]
    dh_array = zeros(img_channel)
    Threads.@threads for i = 1:img_channel
        dh_array[i] = taplineHausdorffDist(imgA[:,:,i], imgB[:,:,i];)
    end
    return dh_array
end

function endpointErr(imgA::Array{Bool,2}, imgB::Array{Bool,2})
    pt_a = findall(imgA);
    pt_b = findall(imgB);

    err_1 = LinearAlgebra.norm((pt_a[1] - pt_b[1]).I) 
    err_2 = LinearAlgebra.norm((pt_a[end] - pt_b[end]).I) 

    return err_1 + err_2
end
function endpointErr(imgA::Array{Bool,3}, imgB::Array{Bool,3})
    img_channel = size(imgA)[3]
    err_array = zeros(img_channel)
    Threads.@threads for i =1:img_channel
        err_array[i] = endpointErr(imgA[:,:,i], imgB[:,:,i];)
    end
    return err_array
end


"""  Bounding Box Eval """
## Bounding Box
#result_model_path = "result/model_1_2/customloss_enable2/"

result_model_path = "result/model_2_2/"
iou_eval = 0.5

ap_box_fold = zeros(Float32, 5)             # average precision for each fold
dh_fold_vector = Vector(undef, 5)           # vector of dh for each fold
eperr_fold_vector = Vector(undef,5)         # vector of endpoint error for each fold
mask_fold_vector = Vector(undef, 5)         # positive detection mask

dh_avg_fold = zeros(Float32, 5)             # avg dh of positive detection
eperr_avg_fold = zeros(Float32, 5)          # avg endpoint error of positive detection

Threads.@threads for fold_no = 1:5
    local dtboxiou
    local dttaplineimg
    local gttapline

    @load "$result_model_path" * "dtboxiou_fold$fold_no.bson" dtboxiou
    @load "$result_model_path" * "dttapline_fold$fold_no.bson" dttaplineimg
    @load "result/model_1_gt/gttapline_fold$fold_no.bson" gttapline

    ap_box_fold[fold_no] = count(x -> x >= iou_eval, dtboxiou) / length(dtboxiou)
    dh_fold_vector[fold_no] = taplineHausdorffDist(gttapline, dttaplineimg)
    eperr_fold_vector[fold_no] = endpointErr(gttapline,dttaplineimg)

    mask_fold_vector[fold_no] =  map(dtboxiou) do x
        if x >= iou_eval
            return 1
        else
            return NaN
        end
    end

    dh_masked = mask_fold_vector[fold_no] .* dh_fold_vector[fold_no]
    dh_avg_fold[fold_no] = Statistics.mean(filter(!isnan, dh_masked))

    eperr_masked = mask_fold_vector[fold_no].* eperr_fold_vector[fold_no]
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