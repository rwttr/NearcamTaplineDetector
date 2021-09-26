# Training 
**Training settings for all models**

- Model Name naming convension: _model_name_ _ _yolo box loss_ _ _yolo objectness loss_ _ _pxl loss_
(std =  standard published YOLO work, MSE for box regression)
- K=5 Fold, 100 Epochs per fold
- Mini-Batch GD w/ Momentum, Base Learning Rate = 0.008, with Warmup, Steady and Decay learning rate scheduler

---
Model Name                  | YOLO Box loss | YOLO Objectness       | Pxl Loss
| :-- | --: | --: | --: |
<br> **model_1v1**
model_1v1_std_std_dice      | MSE           | Tversky, β=0.8        | Dice
model_1v1_std_std_focal     | MSE           | Tversky, β=0.8        | Binary Focal
model_1v1_std_std_tversky   | MSE           | Tversky, β=0.8        | Tversky, β=0.8  
<br> **model_1v2**
model_1v2_std_std_dice      | MSE | Tversky, β=0.8 | Dice
model_1v2_std_std_focal     | MSE | Tversky, β=0.8 | Binary Focal
model_1v2_std_std_tversky   | MSE | Tversky, β=0.8 | Tversky, β=0.8
<br> **model_A**
model_a | MSE | Tversky, β=0.8 | Dice + Binary Focal + <br /> Tversky, β=0.8

$x=1\beta$