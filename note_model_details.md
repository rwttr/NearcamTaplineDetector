# Model Details
## **model_1**
- Input image size : 224 x 224 x 3
- Backbone is Darknet-53 light version (leakyrelu applied in residual block)   
- Box-endpoint included for final detection as starting-terminatin tapping line pixels
    - YOLO box detection layers (tx, ty, tw, th)
        - 2 branches for box detection with tanh output for center-regression
        - 3 anchors
    - U-net style upsampling path (1 branch for edge detection)
        - prediction output size: 224 x 224 x 1 
    

### **model_1 variant 1** (model_1v1)
- U-net prediction header
    - 1x1 pointwise conv with tanh output
    - column-wise softmax for edge thinning (conform nearcam tapline dataset)
```julia
    # header for edge prediction of model_1v1
    pxpredhead_block = Chain(
        Conv((1, 1), 64 => 1, pad=SamePad(), tanh), # pointwise conv
        x -> softmax(x, dims=1) # column-wise softmax
    )
```

### **model_1 variant 2** (model_1v2)
- U-net prediction header
    - 1x1 pointwise conv with standard pixel-wise sigmoid output
```julia
    # header for edge prediction of model_1v2
    pxpredhead_block = Conv((1, 1), 64 => 1, pad=SamePad(), sigmoid)    
```
---

## **model_A**
- Extend **model_1** in edge detection branch
    - 3 parallel detection maps with 1 Column-wise Softmax for final prediction, gather via addition
```julia
    # header for edge prediction of model_a
    pxlsup_1 = Conv((1, 1), 64 => 1, pad=SamePad(), sigmoid)
    pxlsup_2 = Conv((1, 1), 64 => 1, pad=SamePad(), sigmoid)
    pxlsup_3 = Conv((1, 1), 64 => 1, pad=SamePad(), sigmoid)
    
    pxpredhead_block = pxlsup_1, pxlsup_2, pxlsup_3

    # for forward pass function
    pxout = softmax(pxsup_1 + pxsup_2 + pxsup_3, dims=1)
```


