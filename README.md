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

| Experiment name          | CV score        | LB score   | Encoder            | Extra data | Note    | Args |
|--------------------------|-----------------|------------|--------------------|------------|---------|-|
| Baseline classification  | 0.9077 ± 0.0045 | 0.625      | Resnet18           | No         | 4 folds |-|
| Baseline regression      | 0.9093 ± 0.0033 | 0.646      | Resnet18           | No         | 4 folds |-|
|--------------------------|-----------------|------------|--------------------|------------|---------|-|
| Baseline classification  | 0.9213 ± 0.0033 | 0.772      | SEResnext50        | No         | 4 folds |-|
| Baseline regression      | 0.9225 ± 0.0022 | 0.787      | SEResnext50        | No         | 2 folds |-|
| Baseline regression      | 0.9176 ± 0.0080 | 0.763      | SEResnext101       | No         | 4 folds, Multi-pooling |-|
| Baseline classification  | 0.8055 ± 0.0065 | 0.714 (?1) | cls_resnext50_gap  | Yes        | 4 folds |-|
| Baseline regression      | 0.9234 ± 0.0035 | 0.791      | reg_resnext50_rms  | No         | 4 folds |-|
| Baseline regression      | 0.9129 ± 0.0099 | 0.804      | reg_resnext50_rms  | Yes        | 4 folds |-|
| Baseline regression      | 0.9200 ± 0.0044 | 0.803      | reg_resnext50_rms  | Yes        | 4 folds (768) |-|
|--------------------------|-----------------|------------|-----------------------|------------|---------|-|
| Baseline regression      | 0.9128 ± 0.0066 | 0.799      | reg_seresnext50_rms   | Yes        | 4 folds |-|
| Baseline regression      | 0.8992 ± 0.0041 |            | reg_seresnext101_rms  | Yes        | 4 folds |-|
| Baseline regression      | 0.9018 ± 0.0079 | 0.774      | reg_densenet201_rms   | Yes        | 4 folds |-|
| Baseline regression      | 0.9053 ± 0.0053 | 0.761      | reg_inceptionv4_rms   | Yes        | 4 folds |-|
|--------------------------|-----------------|------------|-----------------------|------------|---------|-|
| Regression with aux loss | 0.9170 ± 0.0049 | 0.787      | reg_seresnext50_rms   | Yes        | 4 folds | -m reg_seresnext50_rms -a medium -f 0 -f 1 -f 2 -f 3 -b 60 -l clipped_mse --fp16 -o Adam -d 0.5 -s multistep -lr 1e-4 -wd 1e-4 -e 100 -v --use-idrid --use-messidor --use-aptos2019 --warmup 10 |
|--------------------------|-----------------|------------|-----------------------|------------|---------|-|
| Regression with aux loss | 0.9244 ± 0.0060 | 0.752      | reg_seresnext50_rms   | Aptos2015      | 4 folds | train_regression_baseline.py -m reg_seresnext50_rms -a medium -f 0 -f 1 -f 2 -f 3 -b 60 --fp16 -o Adam -d 0.5 -s multistep -lr 1e-4 -e 100 -es 20 -v --use-aptos2019 --warmup 10
| Regression with aux loss | 0.8737 ± 0.0214 | 0.668      | reg_seresnext50_rms   | IDRID          | 4 folds | train_regression_baseline.py -m reg_seresnext50_rms -a medium -f 0 -f 1 -f 2 -f 3 -b 60 --fp16 -o Adam -d 0.5 -s multistep -lr 1e-4 -e 100 -es 20 -v --use-idrid --warmup 10
| Regression with aux loss | 0.9006 ± 0.0141 | 0.554      | reg_seresnext50_rms   | Messidor       | 4 folds | train_regression_baseline.py -m reg_seresnext50_rms -a medium -f 0 -f 1 -f 2 -f 3 -b 60 --fp16 -o Adam -d 0.5 -s multistep -lr 1e-4 -e 100 -es 20 -v --use-messidor --warmup 10
| Regression with aux loss | 0.9134 ± 0.0044 | 0.779      | reg_seresnext50_rms   | A15, ID, MD    | 4 folds | train_regression_baseline.py -m reg_seresnext50_rms -a medium -f 0 -f 1 -f 2 -f 3 -b 60 --fp16 -o Adam -d 0.5 -s multistep -lr 1e-4 -e 100 -es 20 -v --use-messidor --warmup 10
|--------------------------|-----------------|------------|-----------------------|------------|---------|-|
| Regression with aux loss | 0.9231 ± 0.0043 | 0.813      | seresnextd50_gwap     | A15, ID, MD    | 4 folds | train_regression_baseline.py train_regression_baseline.py -m seresnext50d_gwap -a medium -f 0 -f 1 -f 2 -f 3 -b 60 --size 512 --fp16 -o AdamW -wd 1e-4 -s multistep -lr 3e-4 -e 50 --use-idrid --use-messidor --use-aptos2019 -v --criterion-cls focal_kappa -l1 2e-4
| Regression with aux loss |                 |            | seresnextd50_gap      | A15, ID, MD    | 4 folds | train_regression_baseline.py train_regression_baseline.py -m seresnext50d_gwap -a medium -f 0 -f 1 -f 2 -f 3 -b 60 --size 512 --fp16 -o AdamW -wd 1e-4 -s multistep -lr 3e-4 -e 50 --use-idrid --use-messidor --use-aptos2019 -v --criterion-cls focal_kappa -l1 2e-4


# References

1. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5961805/
1. https://www.slideshare.net/EducacionMolina/diabetic-retinopathy-71884270

# Other 

Train on Aptos 2019

Validation on IDRID + MESSIDOR: 0.6325 / 0.6291
messidor TTA None Mean 0.5497315432994327 std 0.020016714451918216 MeanAvg 0.5370538498582897
idrid TTA None Mean 0.7221154799857236 std 0.02454628058174581 MeanAvg 0.7603479511540748
aptos2019 TTA None Mean 0.9720265955567333 std 0.0024721825621897855 MeanAvg 0.9873314014864271
aptos2015 TTA None Mean 0.471382231210149 std 0.02805589901308936 MeanAvg 0.4984512005316677
