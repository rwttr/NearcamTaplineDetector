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

Threshold detection @ 0.5 IoU
---
Model|AP @0.5 IoU|Avg.Detected Tapline Endpoint L2 Distance (px)| Avg.Detected Tapline Hausdorff Distance (px) | Avg.Detected F1-Score | Avg.Model Tapline Endpoint L2 Distance (px) | Avg.Model Hausdorff Distance(px) | Avg. Model F1-Score |
| :-- | --: | --: | --: | --: | --: | --: | --: |
<br> **model_1v1**
p5_model_1v1_std_std_dice      | 0.96 | 13.37 | 10.02 | 0.3528 | 14.10 | 10.53 | 0.3494
p5_model_1v1_std_std_focal     | 0.98 | 12.65 | 12.44 | 0.3341 | 13.17 | 12.78 | 0.3312
p5_model_1v1_std_std_tversky   | 0.94 | 14.12 | 12.31 | 0.3672 | 15.58 | 13.28 | 0.3672
p5_model_1v1_std_std_all       | - | - | - | - | - | - | -
no_model_1v1_std_std_dice      | 0.98 | 12.08 | 8.94 | 0.3511 | 12.53 | 9.32 | 0.3491
no_model_1v1_std_std_focal     | 0.96 | 12.68 | 12.31 | 0.3142 | 32.38 | 23.11 | 0.2641
no_model_1v1_std_std_tversky   | 0.98 | 12.19 | 11.31 | 0.3510 | 12.55 | 11.60 | 0.3491
no_model_1v1_std_std_all       | - | - | - | - | - | - | -
<br> **model_1v2**
p5_model_1v2_std_std_dice      | 0.98 | 12.00 | 9.99 | 0.5652 | 12.21 | 10.14 | 0.5642
p5_model_1v2_std_std_focal     | 0.97 | 12.77 | 12.50 | 0.3990 | 13.34 | 12.86 | 0.3955
p5_model_1v2_std_std_tversky   | 0.96 | 13.52 | 22.91 | 0.5568 | 14.19 | 23.23 | 0.5521
p5_model_1v2_std_std_all       | - | - | - | - | - | - | -
no_model_1v2_std_std_dice      | 0.98 | 12.60 | 11.18 | 0.5510 | 12.90 | 11.34 | 0.5494
no_model_1v2_std_std_focal     | 0.95 | 13.22 | 13.54 | 0.3890 | 14.47 | 14.28 | 0.3837
no_model_1v2_std_std_tversky   | 0.97 | 12.68 | 23.65 | 0.5602 | 13.50 | 24.04 | 0.5547
no_model_1v2_std_std_all       | - | - | - | - | - | - | -
<br> **model_a**
p5_model_a                     | 0.98 | 11.93 | 10.79 | 0.5629 | 12.22 | 10.99 | 0.5615
no_model_a                     | 0.98 | 12.08 | 10.60 | 0.5654 | 12.33 | 10.82 | 0.5637
---


**Threshold detection @ 0.75 IoU**
---
Model|AP @0.5 IoU|Avg.Detected Tapline Endpoint L2 Distance (px)| Avg.Detected Tapline Hausdorff Distance (px) | Avg.Detected F1-Score | Avg.Model Tapline Endpoint L2 Distance (px) | Avg.Model Hausdorff Distance(px) | Avg. Model F1-Score |
| :-- | --: | --: | --: | --: | --: | --: | --: |
<br> **model_1v1**
p5_model_1v1_std_std_dice      | 0.56 | 9.51 | 6.88 | 0.3640 | - | - | -
p5_model_1v1_std_std_focal     | 0.57 | 9.34 | 10.24 | 0.3455 | - | - | -
p5_model_1v1_std_std_tversky   | 0.48 | 9.93 | 9.58 | 0.3908 | - | - | -
p5_model_1v1_std_std_all       | - | - | - | - | - | - | -
no_model_1v1_std_std_dice      | 0.60 | 8.90 | 6.60 | 0.3614 | - | - | -
no_model_1v1_std_std_focal     | 0.58 | 9.25 | 9.34 | - | - | -
no_model_1v1_std_std_tversky   | 0.61 | 9.15 | 8.69 | 0.3660
no_model_1v1_std_std_all       | - | - | - | - | - | - | -
<br> **model_1v2**
p5_model_1v2_std_std_dice      | 0.62 | 9.17 | 7.28 | 0.5819 | - | - | -
p5_model_1v2_std_std_focal     | 0.57 | 9.35 | 10.16 | 0.4158 | - | - | -
p5_model_1v2_std_std_tversky   | 0.50 | 9.57 | 21.36 | 0.5747 | - | - | -
p5_model_1v2_std_std_all       | - | - | - | - | - | - | -
no_model_1v2_std_std_dice      | 0.58 | 9.20 | 8.19 | 0.5744 | - | - | -
no_model_1v2_std_std_focal     | 0.53 | 9.37 | 11.25 | 0.4028 | - | - | -
no_model_1v2_std_std_tversky   | 0.58 | 9.13 | 22.06 | 0.5844 | - | - | -
no_model_1v2_std_std_all       | - | - | - | - | - | - | -
<br> **model_a**
p5_model_a                     | 0.62 | 8.74 | 7.64 | 0.5810 | - | - | -
no_model_a                     | 0.62 | 9.14 | 7.79 | 0.5831 | - | - | -
---

## Prediction Result for model_a (Voting Output)
- Tapping Line inference in all model performs
    1. Box: merge all box predictions from 2 YOLO branches with NMS, then pick one box with the highest score
    2. Tapping Line: argmax row-position on every column in Pxl prediciton branch (column-wise argmax) ** *separately* **, for model_a is 3 Pxl outputs
    3. Crop all 2. with 1. with add top-left and btm-right box corners
    4. Pixelwise voting for all 3. for majority (2-of-3) to be tapping line output pixel
- Results averged from KFold Cross Validation
- Model Naming Convension
    - *p5* : predicted box with >=0.5 IoU is ignored in loss calculation
    - *no* : all YOLO receptive cell are in loss calculation

---
Model|AP @0.5 IoU|Avg.Detected Tapline Endpoint L2 Distance (px)| Avg.Detected Tapline Hausdorff Distance (px) | Avg.Detected F1-Score | Avg.Model Tapline Endpoint L2 Distance (px) | Avg.Model Hausdorff Distance(px) | Avg. Model F1-Score |
| :-- | --: | --: | --: | --: | --: | --: | --: |
p5_model_a  | 0.98 | 11.93 | 9.00 | 0.5670 | 12.21 | 9.21 | 0.5658
no_model_a  | 0.98 | 12.08 | 9.19 | 0.5685 | 12.33 | 9.39 | 0.5667
---

Model|AP @0.75 IoU|Avg.Detected Tapline Endpoint L2 Distance (px)| Avg.Detected Tapline Hausdorff Distance (px) | Avg.Detected F1-Score | Avg.Model Tapline Endpoint L2 Distance (px) | Avg.Model Hausdorff Distance(px) | Avg. Model F1-Score |
| :-- | --: | --: | --: | --: | --: | --: | --: |
p5_model_a | 0.62 | 8.73 | 6.27 | 0.5843 | - | - | -
no_model_a | 0.62 | 9.14 | 6.77 | 0.5851 | - | - | -
---