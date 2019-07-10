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
    **Question**:How much boots it adds? 

1. Go for more models
    > Train resnet18, resnet34, densenet, efficientnet, senet zoo

### Other

1. Catalyst config

## Results

| Experiment name         | CV score        | LB score | Encoder  |
|-------------------------|-----------------|----------|----------|
| Baseline classification | 0.9077 ± 0.0045 | 0.625    | Resnet18 |
| Baseline regression     | 0.9093 ± 0.0033 | 0.646    | Resnet18 |
