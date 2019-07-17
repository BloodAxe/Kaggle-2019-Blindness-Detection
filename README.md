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

## Results

| Experiment name         | CV score        | LB score | Encoder      | Extra data | Note    |
|-------------------------|-----------------|----------|--------------|------------|---------|
| Baseline classification | 0.9077 ± 0.0045 | 0.625    | Resnet18     | No         | 4 folds |
| Baseline regression     | 0.9093 ± 0.0033 | 0.646    | Resnet18     | No         | 4 folds |
|-------------------------|-----------------|----------|--------------|------------|---------|
| Baseline classification | 0.9213 ± 0.0033 | 0.772    | SEResnext50  | No         | 4 folds |
| Baseline regression     | 0.9225 ± 0.0022 | 0.787    | SEResnext50  | No         | 2 folds |
| Baseline regression     | 0.9176 ± 0.0080 | 0.763    | SEResnext101 | No         | 4 folds, Multi-pooling |

# References

1. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5961805/