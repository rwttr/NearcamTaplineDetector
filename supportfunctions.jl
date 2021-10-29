""" All the Functions that used in this project """
###################################################

""" Network Input Utils """
# normalize input image
function normalizeImg(imgTensor::Array{Float32,4})
    # rescale imgTensor channel values to range 0-1 then
    # apply z-score normalization in each channel
    c, batchsize = size(imgTensor)[3:4];
    Threads.@threads for batch_no = 1:batchsize
        for chan_no = 1:c
            # rescale UnitRangeTransform
            imgTensor[:,:,chan_no,batch_no] = 
                StatsBase.standardize(UnitRangeTransform, imgTensor[:,:,chan_no,batch_no]
            );            
            # z-score
            imgTensor[:,:,chan_no,batch_no] = 
                StatsBase.standardize(ZScoreTransform, imgTensor[:,:,chan_no,batch_no]
            );
        end
    end
    return imgTensor
end

# Input Image Augmentation, Augment: image, bbox, pxlmask 
function augmentTapline(img::Array{Float32,4}, bbox::Array{Float32,4}, pxlmask::Array{Float32,4})
    max_x, max_y, _, batchsize = size(img)
    img_out = similar(img)
    pxl_out = similar(pxlmask)
    box_out = similar(bbox)

    rot_range = -10:0.5:10
    zoom_range = 0.8:0.05:1.2
    
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


""" Boxes Transformation """
# convert bbox, center-xywh to topleft-xywh
function boxCenter2topleft(bbox::NTuple{4,Vector{Float32}})
    boxout = (
        bbox[1] .- bbox[3] ./ 2, 
        bbox[2] .- bbox[4] ./ 2, 
        bbox[3], 
        bbox[4]
    )
    return boxout
end
function boxCenter2topleft(bbox::NTuple{4,CuArray{Float32,1}})
    boxout = (
        bbox[1] .- bbox[3] ./ 2, 
        bbox[2] .- bbox[4] ./ 2, 
        bbox[3], 
        bbox[4]
    )
    return boxout
end

function boxCenter2topleft(bbox::Vector{Float32})
    boxout = [
        bbox[1] - bbox[3] ./ 2, 
        bbox[2] - bbox[4] ./ 2, 
        bbox[3], 
        bbox[4]
    ]
    return boxout
end
function boxCenter2topleft(bbox::Matrix{Float32})
    # boxout = [
    #     bbox[:,1] - bbox[:,3] ./ 2, 
    #     bbox[:,2] - bbox[:,4] ./ 2, 
    #     bbox[:,3], 
    #     bbox[:,4]
    # ]
    
    boxout = [
        bbox[:,1] - bbox[:,3] ./ 2 bbox[:,2] - bbox[:,4] ./ 2 bbox[:,3] bbox[:,4]
    ]
    return Float32.(boxout)
end

# convert bbox, topleft-xywh to center-xywh
function boxTopleft2center(bbox::Vector{Float32})
    boxout = [
        bbox[1] + bbox[3] ./ 2, 
        bbox[2] + bbox[4] ./ 2, 
        bbox[3],
        bbox[4]
    ];
    return boxout
end



""" YOLO box Convention """
# convert [tx, ty, tw, th] (log-scale) -> [bx, by, bw, bh] (spaitial)
function predBxBy(txTensor, tyTensor, yololayer::YoloLayer)
    c, batchsize = size(txTensor)[3:4]
    enlarge_factor = yololayer.featuremap_stride;
    bx = (txTensor + repeat(yololayer.x_grid_offset, 1, 1, c, batchsize)) .* enlarge_factor
    by = (tyTensor + repeat(yololayer.y_grid_offset, 1, 1, c, batchsize)) .* enlarge_factor
    return bx, by
end
function predBwBh(twTensor, thTensor, yololayer::YoloLayer)
    c = size(twTensor)[3]
    bw = copy(twTensor);
    bh = copy(thTensor);
    enlarge_factor = yololayer.featuremap_stride;
    @assert c == yololayer.num_anchors
    for i = 1:c
        bw[:,:,i,:] = yololayer.anchors[i][1] .* exp.(twTensor[:,:,i,:]) .* enlarge_factor
        bh[:,:,i,:] = yololayer.anchors[i][2] .* exp.(thTensor[:,:,i,:]) .* enlarge_factor
    end
    return bw, bh
end

# Intersection Over Union
function bboxIoU(box1::Vector{Float32}, box2::Vector{Float32})
    # box1, box2 are in topleft-xy wh format
    # zero-center box1 and box2
    b1_x1, b1_x2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    b1_y1, b1_y2 = box1[2] - box1[4] / 2, box1[2] + box1[4] / 2
    b2_x1, b2_x2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    b2_y1, b2_y2 = box2[2] - box2[4] / 2, box2[2] + box2[4] / 2

    intersectArea = max(( min(b1_x2, b2_x2) - max(b1_x1, b2_x1) ), 0) * 
        max(( min(b1_y2, b2_y2) - max(b1_y1, b2_y1) ), 0);

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1;
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1;
    unionArea = (w1 * h1) + (w2 * h2) - intersectArea + eps(Float32);

    iou = intersectArea / unionArea;
    return Float32(iou);
end



""" Non-max suppression """
function nmsBox(pred_vec; iou_th=0.5)
    # non-max suppression
    # input = box prediction vector [bx by bw bh objscore]
    batchsize = length(pred_vec)
    box_nms_vec = []

    for batch_no = 1:batchsize
        # initialize
        pred_vec_batch = pred_vec[batch_no]
        b_vec = map(x -> x[1:4], pred_vec_batch)
        b_score = map(x -> x[5], pred_vec_batch)

        # selected box
        d_vec = [];
        while !isempty(b_vec)
            # NMS: select a box with highest score
            # pick from box vector then add to d_vec
            _, pickidx = findmax(b_score);
            pickbox = popat!(b_vec, pickidx);
            popat!(b_score, pickidx);
            push!(d_vec, pickbox);

            # compare iou of selected box to the remaining
            # remove box above threshold
            remove_count = 0;
            for i = 1:length(b_vec)
                iou = bboxIoU(pickbox, b_vec[i - remove_count])
                if iou > iou_th
                    popat!(b_vec, i - remove_count);
                    popat!(b_score, i - remove_count);
                    remove_count += 1; # index shift
                end
            end
        end
        # box_nms_vec[batch_no] = d_vec
        push!(box_nms_vec, d_vec)
    end
    return box_nms_vec
end

# non-max suppression with score output
function nmsBoxwScore(pred_vec; iou_th=0.5)
    # non-max suppression with score output (for test mode ) 
    # input = box prediction vector [bx by bw bh objscore]
    batchsize = length(pred_vec)
    box_nms_vec = Array{Vector}(undef, batchsize)

    for batch_no = 1:batchsize
        # initialize
        pred_vec_batch = pred_vec[batch_no]
        b_vec = map(x -> x[1:5], pred_vec_batch)
        b_score = map(x -> x[5], pred_vec_batch)

        # selected box
        d_vec = []; 

        while !isempty(b_vec)
            # NMS: select a box with highest score
            # pick from box vector then add to d_vec
            _, pickidx = findmax(b_score);
            pickbox = popat!(b_vec, pickidx);
            popat!(b_score, pickidx);
            push!(d_vec, pickbox);

            # compare iou of selected box to the remaining
            # remove box above threshold
            remove_count = 0;
        
            for i = 1:length(b_vec)
                iou = bboxIoU(pickbox, b_vec[i - remove_count])
                if iou > iou_th
                    popat!(b_vec, i - remove_count);
                    popat!(b_score, i - remove_count);
                    remove_count += 1; # index shift
                end
            end
        end
        box_nms_vec[batch_no] = d_vec
    end
    return box_nms_vec
end



""" Translate YOLO head
    transform tensor to spatial domain (detected box) at a yolohead
    transform (tx, ty, tw, th) to (bx, by, bw, bh)
"""
function transYolohead(nnTensorTp::NTuple{6,Array{Float32,4}}, yololayer::YoloLayer)
    txTensor, tyTensor, twTensor, thTensor, objTensor = nnTensorTp[1:5]
    w, h, c, batchsize = size(txTensor)
    enlarge_factor = Float32(yololayer.featuremap_stride)

    # tx, ty: offsetting * enlarge_factor
    local bx_add
    local by_add
    Zygote.ignore() do
        bx_add = repeat(yololayer.x_grid_offset, 1, 1, c, batchsize)
        by_add = repeat(yololayer.y_grid_offset, 1, 1, c, batchsize)
        bx_add = Float32.(bx_add)
        by_add = Float32.(by_add)
    end
    bxTensor = (txTensor + bx_add) .* enlarge_factor
    byTensor = (tyTensor + by_add) .* enlarge_factor

    # tw, th multiplier (constant) (avoid zygote gradient)
    local tw_mul
    local th_mul
    Zygote.ignore() do
        tw_mul = zeros(w, h, c, batchsize)
        th_mul = zeros(w, h, c, batchsize)
        for anchor_no = 1:yololayer.num_anchors
            tw_mul[:,:,anchor_no,:] = repeat([yololayer.anchors[anchor_no][1]], w, h, 1, batchsize)
            th_mul[:,:,anchor_no,:] = repeat([yololayer.anchors[anchor_no][2]], w, h, 1, batchsize)
        end
        tw_mul = Float32.(tw_mul)
        th_mul = Float32.(th_mul)
    end
    bwTensor = exp.(twTensor) .* tw_mul .* enlarge_factor
    bhTensor = exp.(thTensor) .* th_mul .* enlarge_factor

    return (bxTensor, byTensor, bwTensor, bhTensor, objTensor)
end
function transYolohead(nnTensorTp::NTuple{6,CuArray{Float32,4}}, yololayer::YoloLayer)
    txTensor, tyTensor, twTensor, thTensor, objTensor = nnTensorTp[1:5]
    w, h, c, batchsize = size(txTensor)
    enlarge_factor = Float32(yololayer.featuremap_stride) |> gpu

    # tx, ty offsetting * enlarge_factor
    local bx_add
    local by_add
    Zygote.ignore() do
        bx_add = Float32.(repeat(yololayer.x_grid_offset, 1, 1, c, batchsize)) |> gpu
        by_add = Float32.(repeat(yololayer.y_grid_offset, 1, 1, c, batchsize)) |> gpu
    end
    bxTensor = (txTensor + bx_add) .* enlarge_factor
    byTensor = (tyTensor + by_add) .* enlarge_factor

    # tw, th multiplier (constant) (avoid zygote gradient)
    local tw_mul
    local th_mul
    Zygote.ignore() do
        tw_mul = zeros(Float32, w, h, c, batchsize)
        th_mul = zeros(Float32, w, h, c, batchsize)
        for anchor_no = 1:yololayer.num_anchors
            tw_mul[:,:,anchor_no,:] = repeat([yololayer.anchors[anchor_no][1]], w, h, 1, batchsize)
            th_mul[:,:,anchor_no,:] = repeat([yololayer.anchors[anchor_no][2]], w, h, 1, batchsize)
        end
        tw_mul = Float32.(tw_mul) |> gpu
        th_mul = Float32.(th_mul) |> gpu
    end
    bwTensor = exp.(twTensor) .* tw_mul .* enlarge_factor
    bhTensor = exp.(thTensor) .* th_mul .* enlarge_factor

    return (bxTensor, byTensor, bwTensor, bhTensor, objTensor)
end



""" Predict(Detect) Box from YOLO head """
function predictboxYolohead(nnTensorTuple::NTuple{6,Array{Float32,4}}, yololayer::YoloLayer; obj_th=0.5)
    txTensor, tyTensor, twTensor, thTensor, objTensor = nnTensorTuple[1:5]
    bxTensor, byTensor = predBxBy(txTensor, tyTensor, yololayer);
    bwTensor, bhTensor = predBwBh(twTensor, thTensor, yololayer);

    # x : input tensor
    w, h, num_anchors, batchsize = size(bxTensor)
    @assert num_anchors == yololayer.num_anchors

    # prediction result : box vector
    pred_vec = [];

    for batch_no = 1:batchsize
        pred_vec_batch = []; # predicted box at a batch

        for anchor_no = 1:yololayer.num_anchors
            pred_vec_anchor = []; # predicted box at a anchor

            pred_bx = bxTensor[:,:,anchor_no,batch_no]
            pred_by = byTensor[:,:,anchor_no,batch_no]
            pred_bw = bwTensor[:,:,anchor_no,batch_no]
            pred_bh = bhTensor[:,:,anchor_no,batch_no]
            pred_obj = objTensor[:,:,anchor_no,batch_no]

            # find all possible boxes (high-memory loaded)
            for i = 1:(w * h) # eachindex(pred_bx)
                # convert to topleft xywh
                bx, by, bw, bh = boxCenter2topleft(Float32.([pred_bx[i];pred_by[i];pred_bw[i];pred_bh[i]]));
                obj_score = pred_obj[i];

                if (obj_score >= obj_th) &&                             # threshold objectness
                    (bx > 0) && (by > 0) && (bw > 0) && (bh > 0)        # remove invalid prediction
                    push!(pred_vec_anchor, [bx by bw bh obj_score]);                   
                end
            end     

            # pick highest score for each anchor
            if !isempty(pred_vec_anchor)
                pred_score = map(x -> x[5], pred_vec_anchor)
                # find max score box for each anchor
                maxIdx = findmax(pred_score)[2]
                push!(pred_vec_batch, pred_vec_anchor[maxIdx])
            end          
        end
        push!(pred_vec, pred_vec_batch)        
    end    
    return pred_vec
end



""" Post-Process: Masking """
# mask output image with bbox
function maskBoxTapline(nnTensor, bbox)
    # nnTensor = edge map image (w,h,c,b)
    max_y, max_x = size(nnTensor)[1:2]
    bbox = Int.(round.(bbox)) # x y w h : col, row, width, height
    x, y, w, h = bbox
    col_1 = clamp(x, 1, max_x)
    col_2 = clamp((x + w), 1, max_x)
    row_1 = clamp(y, 1, max_y)
    row_2 = clamp((y + h), 1, max_y)

    # mask area outside bbox = 0
    mask = zeros(Float32, max_x, max_y)
    mask[row_1:row_2,col_1:col_2] .= 1;

    mask = mask .* nnTensor
    # add box endpoints
    mask[row_1,col_1,:,:] .= 1;
    mask[row_2,col_2,:,:] .= 1;

    return mask
end

# find max-value pixel in each column
function maskColMax(nnTensor)
    # nnTensor = pixel-classidication output feature map
    imgout = zeros(Float32, size(nnTensor)...)
    taplinepx = findmax(nnTensor, dims=1)[2]
    imgout[taplinepx] .= 1
    return imgout
end



""" Model Prediction """
# predict bounding boxes at a branch (test mode)
function predictBox(nnoutputYolohead, yololayer::YoloLayer; obj_th=0.0, nms_th=0.5)
    # nnoutputYolo : network's output of a yolohead
    # obj_th : objectness confident threshold
    # nms_th : non-max iou threshold
    yoloTranslated = predictboxYolohead(nnoutputYolohead, yololayer, obj_th=obj_th);
    dtbox = nmsBoxwScore(yoloTranslated, iou_th=nms_th);
    return dtbox
end

# predict tapping line within pred_box (test mode)
function predictTapline(nnoutputEdgemap, pred_box)
    batchsize = size(nnoutputEdgemap)[4]
    @assert batchsize == length(pred_box)

    tapline_img_vec = Array{Vector}(undef, batchsize);
    for batch_no = 1:batchsize
        tapline_img_batch = [];
        bbox_batch = pred_box[batch_no]
        colmax_img = maskColMax(nnoutputEdgemap[:,:,:,batch_no])        
        for i = 1:length(pred_box[batch_no])
            tapline_img_temp = maskBoxTapline(colmax_img, bbox_batch[i][1:4])
            push!(tapline_img_batch, tapline_img_temp)
        end    
        tapline_img_vec[batch_no] = tapline_img_batch
    end
    return tapline_img_vec
end



""" Geometry Distance """
# L2- distance between 2 tapping line image
function taplineL2Dist(imgA, imgB)
    # drop singleton dimensions
    imgA_d = dropdims(imgA, dims=(findall(size(imgA) .== 1)...,))
    imgB_d = dropdims(imgB, dims=(findall(size(imgB) .== 1)...,))
    px_a = findall(Bool.(imgA_d));
    px_b = findall(Bool.(imgB_d));

    if isempty(px_a) || isempty(px_b)
        return -1;
    end
    # distance from a to b
    a_b_dist = 0;
    for i = 1:length(px_a)
        a = px_a[i]
        # tempdist = [];
        tempdist = zeros(length(px_b))
        for j = 1:length(px_b)
            b = px_b[j]
            # push!(tempdist, LinearAlgebra.norm((a - b).I))
            tempdist = LinearAlgebra.norm((a - b).I)
        end
        a_b_dist += findmin(tempdist)[1];
    end
    # distance from b to a
    b_a_dist = 0;
    for i = 1:length(px_b)
        b = px_b[i]
        # tempdist = [];
        tempdist = zeros(length(px_a))
        for j = 1:length(px_a)
            a = px_a[j]
            # push!(tempdist, LinearAlgebra.norm((a - b).I))
            tempdist[j] = LinearAlgebra.norm((a - b).I)
        end
        b_a_dist += findmin(tempdist)[1];
    end
    return a_b_dist + b_a_dist;
end

# L2- distance between 2 tapping line image (differentiable)
function taplineL2Dist2(imgA, imgB)
    max_row, max_col = size(imgA)[1:2]
    # drop singleton dimensions
    imgA_d = dropdims(imgA, dims=(findall(size(imgA) .== 1)...,))
    imgB_d = dropdims(imgB, dims=(findall(size(imgB) .== 1)...,))
    px_a = findall(Bool.(imgA_d));
    px_b = findall(Bool.(imgB_d));

    if isempty(px_a) || isempty(px_b)
        return -1;
    end
    # distance from a to b
    a_b_dist = 0
    a_b_min = (max_col + max_row) / 2;
    for a in px_a
        tempdist = 0
        for b in px_b
            tempdist = LinearAlgebra.norm((a - b).I)
            if tempdist < a_b_min
                a_b_min = tempdist
            end
        end
        a_b_dist += a_b_min
    end
    # distance from b to a
    b_a_dist = 0
    b_a_min = (max_col + max_row) / 2;
    for b in px_b
        tempdist = 0
        for a in px_a
            tempdist = LinearAlgebra.norm((a - b).I)
            if tempdist < b_a_min
                a_b_min = tempdist
            end
        end
        b_a_dist += a_b_min;
    end
    return a_b_dist + b_a_dist;
end

# L2- distance between 2 tapping line image (differentiable)
function taplineL2Dist3(imgA, imgB)
    # drop singleton dimensions
    imgA_d = dropdims(imgA, dims=(findall(size(imgA) .== 1)...,))
    imgB_d = dropdims(imgB, dims=(findall(size(imgB) .== 1)...,))

    px_a = findall(Bool.(imgA_d));
    px_b = findall(Bool.(imgB_d));

    px_a_row = map(x -> x.I[1], px_a)
    px_a_col = map(x -> x.I[2], px_a)

    px_b_row = map(x -> x.I[1], px_b)
    px_b_col = map(x -> x.I[2], px_b)

    if isempty(px_a) || isempty(px_b)
        return 0;
    end
    # distance from a to b
    a_b_dist = 0
    for i = 1:length(px_a)
        a_row = px_a_row[i]
        a_col = px_a_col[i]
        diff_pt = tuple.(a_row .- px_b_row, a_col .- px_b_col)
        tempdist = LinearAlgebra.norm.(diff_pt)
        a_b_dist += findmin(tempdist)[1]
    end
    # distance from b to a
    b_a_dist = 0
    for i = 1:length(px_b)
        b_row = px_b_row[i]
        b_col = px_b_col[i]
        diff_pt = tuple.(b_row .- px_a_row, b_col .- px_a_col)
        tempdist = LinearAlgebra.norm.(diff_pt)
        b_a_dist += findmin(tempdist)[1]
    end
    return a_b_dist + b_a_dist;
end

# hausdorff distance between 2 tapping line
function taplineHausdorffDist(imgA, imgB)
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



""" 
    Network training related functions  
"""

""" YOLO Loss Function """
# Standard YOLO : MSE
# Bounding Box Loss : compute loss on tx, ty, tw, th
function yoloBoxLoss(nnTensorTp::NTuple{6,Array{Float32,4}}, targetTensorTp::NTuple{5,Array{Float32,4}}, maskArray)
    # tuple format
    nn_tx, nn_ty, nn_tw, nn_th = nnTensorTp[1:4];
    gt_tx, gt_ty, gt_tw, gt_th = targetTensorTp[1:4];
    @assert size(maskArray) == size(nn_th)

    box_tx_loss = Flux.mse(maskArray .* gt_tx, maskArray .* nn_tx);
    box_ty_loss = Flux.mse(maskArray .* gt_ty, maskArray .* nn_ty);
    box_tw_loss = Flux.mse(maskArray .* gt_tw, maskArray .* nn_tw);
    box_th_loss = Flux.mse(maskArray .* gt_th, maskArray .* nn_th);

    return  box_tx_loss + box_ty_loss + box_tw_loss + box_th_loss;
end
function yoloBoxLoss(nnTensorTp::NTuple{6,CuArray{Float32,4}}, targetTensorTp::NTuple{5,CuArray{Float32,4}}, maskArray)
    # tuple format
    nn_tx, nn_ty, nn_tw, nn_th = nnTensorTp[1:4];
    gt_tx, gt_ty, gt_tw, gt_th = targetTensorTp[1:4];
    @assert size(maskArray) == size(nn_th)

    box_tx_loss = Flux.mse(maskArray .* gt_tx, maskArray .* nn_tx);
    box_ty_loss = Flux.mse(maskArray .* gt_ty, maskArray .* nn_ty);
    box_tw_loss = Flux.mse(maskArray .* gt_tw, maskArray .* nn_tw);
    box_th_loss = Flux.mse(maskArray .* gt_th, maskArray .* nn_th);

    return  box_tx_loss + box_ty_loss + box_tw_loss + box_th_loss;
end

# Smooth L1 (Huber Loss, del = cutoff point = 0.1)
function yoloBoxLoss_smoothL1(nnTensorTp::NTuple{6,CuArray{Float32,4}}, 
    targetTensorTp::NTuple{5,CuArray{Float32,4}}, maskArray, del = 0.1)
    # tuple format
    nn_tx, nn_ty, nn_tw, nn_th = nnTensorTp[1:4];
    gt_tx, gt_ty, gt_tw, gt_th = targetTensorTp[1:4];

    box_tx_loss = Flux.huber_loss(maskArray .* gt_tx, maskArray .* nn_tx, δ = del);
    box_ty_loss = Flux.huber_loss(maskArray .* gt_ty, maskArray .* nn_ty, δ = del);
    box_tw_loss = Flux.huber_loss(maskArray .* gt_tw, maskArray .* nn_tw, δ = del);
    box_th_loss = Flux.huber_loss(maskArray .* gt_th, maskArray .* nn_th, δ = del);

    return  box_tx_loss + box_ty_loss + box_tw_loss + box_th_loss;
end


# Objectness Loss at a YoloLayer
function yoloObjLoss(nnTensorTp::NTuple{6,Array{Float32,4}}, targetTensorTp::NTuple{5,Array{Float32,4}}, maskArray=1f0, β=0.8f0)
    nnTensor = nnTensorTp[5]
    targetTensor = targetTensorTp[5]
    maskedtarget = maskArray .* targetTensor
    maskednn = maskArray .* nnTensor
    return Flux.tversky_loss(maskedtarget, maskednn, β=β)
end
function yoloObjLoss(nnTensorTp::NTuple{6,CuArray{Float32,4}}, targetTensorTp::NTuple{5,CuArray{Float32,4}}, maskArray=1f0, β=0.8f0)
    nnTensor = nnTensorTp[5]
    targetTensor = targetTensorTp[5]
    maskedtarget = gpu(maskArray) .* targetTensor
    maskednn = gpu(maskArray) .* nnTensor
    return Flux.tversky_loss(maskedtarget, maskednn, β=β)
end
function yoloObjLossWbCrossentropy(nnTensorTp::NTuple{6,Array{Float32,4}}, targetTensorTp::NTuple{5,Array{Float32,4}}, maskArray=1f0, β=0.8f0)
    nnTensor = nnTensorTp[5]
    targetTensor = targetTensorTp[5]
    maskedtarget = maskArray .* targetTensor
    maskednn = maskArray .* nnTensor
    return Flux.tversky_loss(maskedtarget, maskednn, β=β)
end



""" Pxl Loss Functions """ 
# UNet- Pixel Branch Loss Functions 
# Pxl Loss functions; pre init bias term
function wbcrossentropy(ŷ, y; ϵ=eps(Float32), β=0.8f0)
    # agg(@.(-y * β * log(ŷ + ϵ) - (1 - β) * (1 - y) * log(1 - ŷ + ϵ)))
    Statistics.mean(@.(-y * β * log(ŷ + ϵ) - (1 - β) * (1 - y) * log(1 - ŷ + ϵ)))
end

function pxlLoss_tversky(nnTensor, targetTensor; β=0.8f0)
    return Flux.tversky_loss(nnTensor, targetTensor, β=β);    
end

function pxlLoss_dice(nnTensor, targetTensor)
    return Flux.dice_coeff_loss(nnTensor, targetTensor);    
end

function pxlLoss_binarycrossentropy(nnTensor, targetTensor)
    return Flux.Losses.binarycrossentropy(nnTensor, targetTensor)
end

function pxlLoss_focal(nnTensor, targetTensor)
    return Flux.Losses.binary_focal_loss(nnTensor, targetTensor)
end



"""  Utility Functions for Training """
# generate a target for a YoloLayer
function generateYolotarget(gtbox::Array{Float32,4}, yololayer::YoloLayer)
    # output size : W * H * yololayer.num_anchors * batchsize;
    # output format tuple 
    # (target_tx, target_ty, target_tw, target_th, target_obj)
    w = h = yololayer.featuremap_size;
    c = yololayer.num_anchors;
    batchsize = size(gtbox)[4];

    # initialize target tensors
    target_tx = zeros(Float32, w, h, c, batchsize);
    target_ty = zeros(Float32, w, h, c, batchsize);
    target_tw = zeros(Float32, w, h, c, batchsize);
    target_th = zeros(Float32, w, h, c, batchsize);
    target_obj = zeros(Float32, w, h, c, batchsize);

    for batch_no = 1:batchsize        
        gtbox_batch = gtbox[:,:,:,batch_no];
        # find which anchors is the most overlopped to the ground truth box
        # --normalize box to featuremap size
        gtbox_batch = gtbox_batch ./ (yololayer.featuremap_stride);
        anchor_wh = yololayer.anchors;

        # convert topleft-xy gtbox to zero-center-xy gtbox
        gtbox_batch_0 = 
        [- gtbox_batch[3] / 2, - gtbox_batch[4] / 2, gtbox_batch[3], gtbox_batch[4]] .|> Float32;

        # box in center-xy format, positioned within featuremap grid 
        gtbox_batch_centerxy = [
            gtbox_batch[1] + (gtbox_batch[3] / 2),
            gtbox_batch[2] + (gtbox_batch[4] / 2),
            gtbox_batch[3],
            gtbox_batch[4]
        ] .|> Float32;

        # overlapped ratio between gtbox and anchors (prior)
        overlap_pr_gt = zeros(Float32, yololayer.num_anchors);

        for i = 1:yololayer.num_anchors
            prbox = vec([gtbox_batch_0[1] gtbox_batch_0[2] anchor_wh[i]]);
            overlap_pr_gt[i] = bboxIoU(gtbox_batch_0, prbox);       
        end
        
        _, bestAnchorIndex = findmax(overlap_pr_gt);              
        active_anchor_wh = anchor_wh[bestAnchorIndex];

        # calculate active box center in featuremap grid
        gt_gridx = round(gtbox_batch_centerxy[1]); 
        gt_gridy = round(gtbox_batch_centerxy[2]);

        # non-zero grid position (1 to featuremap_size)
        gt_gridx = Int.(clamp(gt_gridx, 1, w));
        gt_gridy = Int.(clamp(gt_gridy, 1, h));

        # calculate shift distance to active box center
        gt_tx = gtbox_batch_centerxy[1] - gt_gridx;
        gt_ty = gtbox_batch_centerxy[2] - gt_gridy;

        # calculate anchor width,height adjustment to match gtbox_batch
        gt_tw = log(gtbox_batch_centerxy[3] / active_anchor_wh[1]);
        gt_th = log(gtbox_batch_centerxy[4] / active_anchor_wh[2]);

        # store result to target tensor
        # set target only for active anchor channel 
        target_tx[gt_gridy, gt_gridx, bestAnchorIndex, batch_no] = Float32(gt_tx);
        target_ty[gt_gridy, gt_gridx, bestAnchorIndex, batch_no] = Float32(gt_ty);
        target_tw[gt_gridy, gt_gridx, bestAnchorIndex, batch_no] = Float32(gt_tw);
        target_th[gt_gridy, gt_gridx, bestAnchorIndex, batch_no] = Float32(gt_th);
        target_obj[gt_gridy, gt_gridx, bestAnchorIndex, batch_no] = 1f0;          
    end
    return target_tx, target_ty, target_tw, target_th, target_obj;
end

# translate output and check predicted box overlap to gtbox at a YoloLayer
function predBoxOverlap(nnTensorTp::NTuple{6,Array{Float32,4}}, gtbox::Array{Float32,4}, yololayer::YoloLayer)
    # return box_iou between gtbox and anchors 
    # output array size : w*h*num_anchors*batchsize 
    nn_tx, nn_ty, nn_tw, nn_th = nnTensorTp[1:4];
    w, h, c, batchsize = size(nn_tx);
    @assert c == yololayer.num_anchors
    @assert batchsize == size(gtbox)[4];

    pred_iou = zeros(Float32, w, h, c, batchsize);

    for batch_no = 1:batchsize
        # resize gtbox to featuremap size
        gtbox_batch = gtbox[:,:,:,batch_no] ./ Float32(yololayer.featuremap_stride);        
        anchors_wh = yololayer.anchors ./ yololayer.featuremap_size;
        
        for i = 1:yololayer.num_anchors 
            # predicted box (center-offset and log-scale)
            pred_tx = nn_tx[:,:,i,batch_no];   # center shift-x
            pred_ty = nn_ty[:,:,i,batch_no];   # center shift-y
            pred_tw = nn_tw[:,:,i,batch_no];   # width in log scale related to anchors
            pred_th = nn_th[:,:,i,batch_no];   # height in log scale related to anchors
            # convert to spatial domain [bx, by, bw, bh]
            # --Add the grid offsets to the predicted box center cordinates.
            pred_bx = pred_tx + repeat(yololayer.x_grid_offset, 1, 1);
            pred_by = pred_ty + repeat(yololayer.y_grid_offset, 1, 1);
            # --calculate pred bw, bh from anchor
            pred_bw = anchors_wh[i][1] .* exp.(pred_tw);
            pred_bh = anchors_wh[i][2] .* exp.(pred_th);
            # Convert from center-xy coordinate to topleft-xy box.
            pred_bx_tl = pred_bx .- (pred_bw ./ 2);
            pred_by_tl = pred_by .- (pred_bh ./ 2);
            
            pred_iou_anchor = zeros(Float32, w, h);

            for j in eachindex(pred_iou_anchor)             
                pred_box = Float32.([pred_bx_tl[j], pred_by_tl[j], pred_bw[j], pred_bh[j]]);
                pred_iou_anchor[j] = bboxIoU(vec(gtbox_batch), pred_box);
            end            
            global pred_iou[:,:,i,batch_no] = pred_iou_anchor;
        end
    end  
    return pred_iou;
end

# mask iouTensor cell with mask_iou_threshold
function maskPredBoxOverlap(iouTensor::Array{Float32,4}; mask_iou_threshold=0.5)
    iou_mask = ones(Float32, size(iouTensor));
    for i in eachindex(iou_mask)
        if iouTensor[i] >= mask_iou_threshold
            iou_mask[i] = 0f0;
        end
    end
    return iou_mask;
end

# Learning Rate Scheduler with warm-up and decay 
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

# L2 Regularisation on Generic model m
L2reg(m) = sum(sum(p.^2) for p in Flux.params(m))
L2reg(m::Flux.Params) = sum(sum(p.^2) for p in m)

# Image Visualizer
function convert2RGB(x::Arrya{Float32,3})
    # data in x ∈ [0,1]
    # remap value to Normed(UInt8)

    x_uint= convert(Array{N0f8,3},x) #

    return colorview(RGB,x_uint[:,:,1], x_uint[:,:,2], x_uint[:,:,3])
end