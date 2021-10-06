# Inference Results
## Prediction Result for model_1v1, model_1v2, and model_a (sum output)
- Tapping Line inference in all model performs
    1. Box: merge all box predictions from 2 YOLO branches with NMS, then pick one box with the highest score
    2. Tapping Line: argmax row-position on every column in Pxl prediciton branch (column-wise argmax)
    3. Crop 2. with 1. with add top-left and btm-right box corners
- Results averged from KFold Cross Validation
- Model Naming Convension
    - *p5* : predicted box with >=0.5 IoU is ignored in loss calculation
    - *no* : all YOLO receptive cell are in loss calculation

### Threshold detection @ 0.5 IoU
---
Model|AP @0.5 IoU|Avg.Endpoint Dist.Error(pixels)| Avg.Hausdorff Dist.(pixels)
| :-- | --: | --: | --: |
<br> **model_1v1**
p5_model_1v1_std_std_dice      | 0 | 1 | 2
p5_model_1v1_std_std_focal     | 0 | 1 | 2
p5_model_1v1_std_std_tversky   | 0 | 1 | 2
p5_model_1v1_std_std_all       | 0 | 1 | 2
no_model_1v1_std_std_dice      | 0 | 1 | 2
no_model_1v1_std_std_focal     | 0 | 1 | 2
no_model_1v1_std_std_tversky   | 0 | 1 | 2
no_model_1v1_std_std_all       | 0 | 1 | 2
<br> **model_1v2**
p5_model_1v2_std_std_dice      | 0 | 1 | 2
p5_model_1v2_std_std_focal     | 0 | 1 | 2
p5_model_1v2_std_std_tversky   | 0 | 1 | 2
p5_model_1v2_std_std_all       | 0 | 1 | 2
no_model_1v2_std_std_dice      | 0 | 1 | 2
no_model_1v2_std_std_focal     | 0 | 1 | 2
no_model_1v2_std_std_tversky   | 0 | 1 | 2
no_model_1v2_std_std_all       | 0 | 1 | 2
<br> **model_a**
p5_model_a                      | 0 | 1 | 2
no_model_a                      | 0 | 1 | 2
---
---

### **Threshold detection @ 0.75 IoU**
---
Model|AP @0.75 IoU|Avg.Endpoint Dist.Error (pixels)| Avg.Hausdorff Dist. (pixels)
| :-- | --: | --: | --: |
<br> **model_1v1**
p5_model_1v1_std_std_dice      | 0 | 1 | 2
p5_model_1v1_std_std_focal     | 0 | 1 | 2
p5_model_1v1_std_std_tversky   | 0 | 1 | 2
p5_model_1v1_std_std_all       | 0 | 1 | 2
no_model_1v1_std_std_dice      | 0 | 1 | 2
no_model_1v1_std_std_focal     | 0 | 1 | 2
no_model_1v1_std_std_tversky   | 0 | 1 | 2
no_model_1v1_std_std_all       | 0 | 1 | 2
<br> **model_1v2**
p5_model_1v2_std_std_dice      | 0 | 1 | 2
p5_model_1v2_std_std_focal     | 0 | 1 | 2
p5_model_1v2_std_std_tversky   | 0 | 1 | 2
p5_model_1v2_std_std_all       | 0 | 1 | 2
no_model_1v2_std_std_dice      | 0 | 1 | 2
no_model_1v2_std_std_focal     | 0 | 1 | 2
no_model_1v2_std_std_tversky   | 0 | 1 | 2
no_model_1v2_std_std_all       | 0 | 1 | 2
<br> **model_a**
p5_model_a                      | 0 | 1 | 2
no_model_a                      | 0 | 1 | 2
---

## Prediction Result for model_a (Voting Output)
- Tapping Line inference in all model performs
    1. Box: merge all box predictions from 2 YOLO branches with NMS, then pick one box with the highest score
    2. Tapping Line: argmax row-position on every column in Pxl prediciton branch (column-wise argmax) ** *separately* **, for model_a is 3 Pxl outputs
    3. Crop all 2. with 1. with add top-left and btm-right box corners
    4. Vote all 3. for majority (2-of-3) to be tapping line output
- Results averged from KFold Cross Validation
- Model Naming Convension
    - *p5* : predicted box with >=0.5 IoU is ignored in loss calculation
    - *no* : all YOLO receptive cell are in loss calculation

---
Model|AP @0.5 IoU|Avg.Endpoint Dist.Error (pixels)| Avg.Hausdorff Dist. (pixels)
| :-- | --: | --: | --: |
p5_model_a                      | 0 | 1 | 2
no_model_a                      | 0 | 1 | 2
---

Model|AP @0.75 IoU|Avg.Endpoint Dist.Error (pixels)| Avg.Hausdorff Dist. (pixels)
| :-- | --: | --: | --: |
p5_model_a                      | 0 | 1 | 2
no_model_a                      | 0 | 1 | 2
---