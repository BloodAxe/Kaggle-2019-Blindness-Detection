# Kaggle-2019-Blindness-Detection

## Plan

1. Baseline training
Implement baseline training, with vanilla dataset, target metric, submission code.
Using only one model **seresnext50**.

1. Adversarial validation
    > Run adversarial validation to split folds based on train/test distribution. 
    **Question**:Can it improve CV/LB corellation? 

## Data

1. Add external datasets for training
    > **Question**:How much boots it adds? 


### Fine-tuning
1. Metric learning
    > Train additional head for regression
    **Question**:How much boots it adds? 

1. FocalLoss

### Training
1. Try batch gradients accumulation
    > Do not decrease LR, increase batch size

1. Try SWA optimizer
    **Question**:How much boots it adds? 

1. Custom loss function
    > Write loss function to respect the distance in predictions
      **Question**:How much boots it adds? 

### Models

1. Spatial transformer network
    > Can model automatically learn to crop image?
    **Q**: How much boots it adds? 
    **A**: 

1. Go for more models
    > Train resnet18, resnet34, densenet, efficientnet, senet zoo

### Other

1. Catalyst config
1. DestroyImage augmentation to make image totally useless (very blurry, dark)
2. Augmentation to add retinopathy signs on image
3. Augmentation to crop image from square to rect aspect and back 

## Results

| Experiment name         | CV score        | LB score   | Encoder            | Extra data | Note    |
|-------------------------|-----------------|------------|--------------------|------------|---------|
| Baseline classification | 0.9077 ± 0.0045 | 0.625      | Resnet18           | No         | 4 folds |
| Baseline regression     | 0.9093 ± 0.0033 | 0.646      | Resnet18           | No         | 4 folds |
|-------------------------|-----------------|------------|--------------------|------------|---------|
| Baseline classification | 0.9213 ± 0.0033 | 0.772      | SEResnext50        | No         | 4 folds |
| Baseline regression     | 0.9225 ± 0.0022 | 0.787      | SEResnext50        | No         | 2 folds |
| Baseline regression     | 0.9176 ± 0.0080 | 0.763      | SEResnext101       | No         | 4 folds, Multi-pooling |
| Baseline classification | 0.8055 ± 0.0065 | 0.714 (?1) | cls_resnext50_gap  | Yes        | 4 folds |
| Baseline regression     | 0.9234 ± 0.0035 | 0.791      | reg_resnext50_rms  | No         | 4 folds |
| Baseline regression     | 0.9129 ± 0.0099 | 0.804      | reg_resnext50_rms  | Yes        | 4 folds |
| Baseline regression     | 0.9200 ± 0.0044 | 0.803      | reg_resnext50_rms  | Yes        | 4 folds (768) |
|-------------------------|-----------------|------------|-----------------------|------------|---------|
| Baseline regression     | 0.9128 ± 0.0066 |            | reg_seresnext50_rms   | Yes        | 4 folds |
| Baseline regression     | 0.8992 ± 0.0041 |            | reg_seresnext101_rms  | Yes        | 4 folds |
| Baseline regression     |                 |            | reg_densenet169_rms   | Yes        | 4 folds |
| Baseline regression     | 0.9053 ± 0.0053 |            | reg_inceptionv4_rms   | Yes        | 4 folds |
|-------------------------|-----------------|------------|-----------------------|------------|---------|

# References

1. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5961805/

Train on Aptos 2019

Validation on IDRID + MESSIDOR: 0.6325 / 0.6291
messidor TTA None Mean 0.5497315432994327 std 0.020016714451918216 MeanAvg 0.5370538498582897
idrid TTA None Mean 0.7221154799857236 std 0.02454628058174581 MeanAvg 0.7603479511540748
aptos2019 TTA None Mean 0.9720265955567333 std 0.0024721825621897855 MeanAvg 0.9873314014864271
aptos2015 TTA None Mean 0.471382231210149 std 0.02805589901308936 MeanAvg 0.4984512005316677
