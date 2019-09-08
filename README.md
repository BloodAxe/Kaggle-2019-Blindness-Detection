# Kaggle-2019-Blindness-Detection

10th-place solution code for https://www.kaggle.com/c/aptos2019-blindness-detection/overview.
This repository contains my solution code and provided as is.

**This repository is not maintained**

## Dependencies

```bash
pip install --quiet -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex
pip install git+https://github.com/mapillary/inplace_abn.git@v1.0.3
```

## Results

| Experiment name          | CV score        | LB score   | Encoder               | Extra data     | Note    | Args |
|--------------------------|-----------------|------------|-----------------------|----------------|---------|-|
| Baseline classification  | 0.9077 ± 0.0045 | 0.625      | Resnet18              | No             | 4 folds |-|
| Baseline regression      | 0.9093 ± 0.0033 | 0.646      | Resnet18              | No             | 4 folds |-|
|--------------------------|-----------------|------------|-----------------------|----------------|---------|-|
| Baseline classification  | 0.9213 ± 0.0033 | 0.772      | SEResnext50           | No             | 4 folds |-|
| Baseline regression      | 0.9225 ± 0.0022 | 0.787      | SEResnext50           | No             | 2 folds |-|
| Baseline regression      | 0.9176 ± 0.0080 | 0.763      | SEResnext101          | No             | 4 folds, Multi-pooling |-|
| Baseline classification  | 0.8055 ± 0.0065 | 0.714 (?1) | cls_resnext50_gap     | Yes            | 4 folds |-|
| Baseline regression      | 0.9234 ± 0.0035 | 0.791      | reg_resnext50_rms     | No             | 4 folds |-|
| Baseline regression      | 0.9129 ± 0.0099 | 0.804      | reg_resnext50_rms     | Yes            | 4 folds |-|
| Baseline regression      | 0.9200 ± 0.0044 | 0.803      | reg_resnext50_rms     | Yes            | 4 folds (768) |-|
|--------------------------|-----------------|------------|-----------------------|----------------|---------|-|
| Baseline regression      | 0.9128 ± 0.0066 | 0.799      | reg_seresnext50_rms   | Yes            | 4 folds |-|
| Baseline regression      | 0.8992 ± 0.0041 |            | reg_seresnext101_rms  | Yes            | 4 folds |-|
| Baseline regression      | 0.9018 ± 0.0079 | 0.774      | reg_densenet201_rms   | Yes            | 4 folds |-|
| Baseline regression      | 0.9053 ± 0.0053 | 0.761      | reg_inceptionv4_rms   | Yes            | 4 folds |-|
|--------------------------|-----------------|------------|-----------------------|----------------|---------|-|
| Regression with aux loss | 0.9170 ± 0.0049 | 0.787      | reg_seresnext50_rms   | Yes            | 4 folds | -m reg_seresnext50_rms -a medium -f 0 -f 1 -f 2 -f 3 -b 60 -l clipped_mse --fp16 -o Adam -d 0.5 -s multistep -lr 1e-4 -wd 1e-4 -e 100 -v --use-idrid --use-messidor --use-aptos2019 --warmup 10 |
|--------------------------|-----------------|------------|-----------------------|----------------|---------|-|
| Regression with aux loss | 0.9244 ± 0.0060 | 0.752      | reg_seresnext50_rms   | Aptos2015      | 4 folds | train_regression_baseline.py -m reg_seresnext50_rms -a medium -f 0 -f 1 -f 2 -f 3 -b 60 --fp16 -o Adam -d 0.5 -s multistep -lr 1e-4 -e 100 -es 20 -v --use-aptos2019 --warmup 10
| Regression with aux loss | 0.8737 ± 0.0214 | 0.668      | reg_seresnext50_rms   | IDRID          | 4 folds | train_regression_baseline.py -m reg_seresnext50_rms -a medium -f 0 -f 1 -f 2 -f 3 -b 60 --fp16 -o Adam -d 0.5 -s multistep -lr 1e-4 -e 100 -es 20 -v --use-idrid --warmup 10
| Regression with aux loss | 0.9006 ± 0.0141 | 0.554      | reg_seresnext50_rms   | Messidor       | 4 folds | train_regression_baseline.py -m reg_seresnext50_rms -a medium -f 0 -f 1 -f 2 -f 3 -b 60 --fp16 -o Adam -d 0.5 -s multistep -lr 1e-4 -e 100 -es 20 -v --use-messidor --warmup 10
| Regression with aux loss | 0.9134 ± 0.0044 | 0.779      | reg_seresnext50_rms   | A15, ID, MD    | 4 folds | train_regression_baseline.py -m reg_seresnext50_rms -a medium -f 0 -f 1 -f 2 -f 3 -b 60 --fp16 -o Adam -d 0.5 -s multistep -lr 1e-4 -e 100 -es 20 -v --use-messidor --warmup 10
|--------------------------|-----------------|------------|-----------------------|----------------|---------|-|
| Regression with aux loss | 0.9231 ± 0.0043 | 0.813      | seresnextd50_gwap     | A15, ID, MD    | 4 folds | train_regression_baseline.py -m seresnext50d_gwap -a medium -f 0 -f 1 -f 2 -f 3 -b 60 --size 512 --fp16 -o AdamW -wd 1e-4 -s multistep -lr 3e-4 -e 50 --use-idrid --use-messidor --use-aptos2019 -v --criterion-cls focal_kappa -l1 2e-4
| Regression with aux loss | 0.9216 ± 0.0035 | 0.809      | seresnextd50_gap      | A15, ID, MD    | 4 folds | train_regression_baseline.py -m seresnext50d_gap  -a medium -f 0 -f 1 -f 2 -f 3 -b 60 --size 512 --fp16 -o AdamW -wd 1e-4 -s multistep -lr 3e-4 -e 50 --use-idrid --use-messidor --use-aptos2019 -v --criterion-cls focal_kappa -l1 2e-4
|--------------------------|-----------------|------------|-----------------------|----------------|---------|-|
| Finetuned 806_813        | 0.9680 ± 0.0037 | 0.803      |
| zen_golick   | 0.9142 ± 0.0049 | 0.770      |
| zen_golick_T | 0.9419 ± 0.0066 | 0.784      |
|--------------------------|-----------------|------------|-----------------------|----------------|---------|-|

|--------------------------------|-----------------|------------|-----------------------|----------------|---------|-|
| happy_shirley      | 0.8148 ± 0.0020 | 0.758
| eager_wright       |                 | 0.771
| goofy_heyrovsky    |                 | 0.785
|--------------------------------|-----------------|------------|-----------------------|----------------|---------|-|


|--------------------------------------------------|------------|-----------------------|----------------|---------|-|
| ablation_study_seresnext50d_max_hard       | 0.782      |  
| ablation_study_seresnext50d_max_medium     | 0.787      |  
| ablation_study_seresnext50d_max_light      | 0.765      |  
|--------------------------------------------------|------------|-----------------------|----------------|---------|-|
| ablation_study_seresnext50d_gwap_hard      | 0.778      | 
| ablation_study_seresnext50d_gwap_medium    | 0.794      |  
| ablation_study_seresnext50d_gwap_light     | 0.795      | 
|--------------------------------------------------|------------|-----------------------|----------------|---------|-|
| ablation_study_seresnext50d_gap_hard       | 0.797      |
| ablation_study_seresnext50d_gap_medium     | 0.802      |
| ablation_study_seresnext50d_gap_light      | 0.801      |
|--------------------------------------------------|------------|-----------------------|----------------|---------|-|
| ablation_study_seresnext50d_gwap_uda_medium| 0.787      |
| ablation_study_seresnext50d_rank_medium    | 0.765      |
|--------------------------------------------------|------------|-----------------------|----------------|---------|-|
| youthful_mccarthy                    | 0.791      | 0.9045 ± 0.0042 |
|--------------------------------------------------|------------|-----------------------|----------------|---------|-|
| heuristic_sinoussi (seresnext50)     | 0.823      | 0.9162 ± 0.0055 | 
| heuristic_sinoussi_finetune          | 0.822      | 0.9170 ± 0.0055 | 
|--------------------------------------------------|------------|-----------------------|----------------|---------|-|
| modest_williams                                  | 0.828      | 0.9198 ± 0.0032 | 
| modest_williams (finetune)                       | 0.828      | 0.9204 ± 0.0032 | 
| modest_williams + heuristic_sinoussi             | 0.832      |                       | 
|--------------------------------------------------|------------|-----------------------|----------------|---------|-|
| modest_williams + heuristic_sinoussi             | 0.834      |                       | optimized thresholds on OOF [0.52704163, 1.47657166, 2.42753601, 3.3937439]
| modest_williams + heuristic_sinoussi             | 0.836      |                       | optimized thresholds on Aptos15? [0.52704163, 1.47657166, 2.42753601, 3.3937439 ]
| modest_williams + heuristic_sinoussi             | 0.836      |                       | optimized thresholds on Aptos15? [0.52704163, 1.47657166, 2.42753601, 3.3937439 ]
|--------------------------------------------------|------------|-----------------------|----------------|---------|-|
| inceptionv4_gap_happy_wright         | 0.827      | 0.9154 ± 0.0044
| inceptionv4_gwap_cranky_torvalds     | 0.824      | 0.9066 ± 0.0053
| seresnext50_rnn_clever_roentgen      | 0.831      | 0.9118 ± 0.0057
|--------------------------------------------------|------------|-----------------------|----------------|---------|-|
| modest_williams+heuristic_sinoussi+happy_wright  | 0.843      | 
| modest_williams+heuristic_sinoussi+happy_wright  | 0.844      | FlipLR
| modest_williams+heuristic_sinoussi+happy_wright  | 0.844      | Flip4, mean    | 
| modest_williams+heuristic_sinoussi+happy_wright  | 0.844      | FlipLR, median | 
| modest_williams+heuristic_sinoussi+happy_wright  | 0.842      | Truncated mean
|--------------------------------------------------|------------|-----------------------|----------------|---------|-|
| heuristic_sinoussi+modest_williams++happy_wright+clever_roentgen  |      | 
| heuristic_sinoussi+modest_williams++happy_wright+clever_roentgen  | 0.802      | logistic_regression
| seresnet152-sad-ardinghelli                           | 0.816      | 
| happy_wright                                                      | 0.802      | 
| seresnext101_fpn_512_practical_wright                 | 806        | 0.9142 0.0064 | warmup FPN from
| resnet34_gap_jovial_turing                            | 809        |  0.9088 0.0043
|--------------------------------------------------|------------|-----------------------|----------------|---------|-|
|inceptionv4_gap_512_medium_aptos2019_messidor_idrid_pl1_fold0_wizardly_mestorf_main.pth | 823
|inceptionv4_gap_512_medium_aptos2019_messidor_idrid_pl1_fold0_wizardly_mestorf_warmup.pth |841
|seresnext50_gap_512_medium_fold0_vibrant_johnson.pth | 825 |
|seresnext50_gap_512_medium_fold0_vibrant_johnson.pth | 822 | Non-optimal thresholds
|--------------------------------------------------|------------|-----------------------|----------------|---------|-|
| inceptionv4_gap_512_medium_pl1_epic_shaw | 843 | 0.9747 ± 0.0002 | warmup
| inceptionv4_gap_512_medium_pl1_epic_shaw | 830 | ???             | main (fold 1,2,3)
| seresnext101_gap_pl1_sad_neumann         | 829 | 0.9747 ± 0.0002
| modest_williams+heuristic_sinoussi+happy_wright+shaw |     |
| inceptionv4_gap_512_medium_pl1_epic_shaw |     |              | warmup, normal + clahe


# Extra data
1. http://www.it.lut.fi/project/imageret/diaretdb1_v2_1/

# References

1. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5961805/
1. https://www.slideshare.net/EducacionMolina/diabetic-retinopathy-71884270
1. http://defauw.ai/diabetic-retinopathy-detection/
1. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4944099/
1. http://eyesteve.com/diabetic-retinopathy-grading/
1. http://eyesteve.com/diabetic-retinopathy-grading/
1. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0139148
1. https://mlwave.com/kaggle-ensembling-guide/
1. https://www.kaggle.com/amrmahmoud123/1-guide-to-ensembling-methods
1. http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/
1. https://github.com/MLWave/Kaggle-Ensemble-Guide/blob/master/src/correlations.py
1. https://jamanetwork.com/journals/jamaophthalmology/fullarticle/2734990

# Other 

Train on Aptos 2019

Validation on IDRID + MESSIDOR: 0.6325 / 0.6291
messidor TTA None Mean 0.5497315432994327 std 0.020016714451918216 MeanAvg 0.5370538498582897
idrid TTA None Mean 0.7221154799857236 std 0.02454628058174581 MeanAvg 0.7603479511540748
aptos2019 TTA None Mean 0.9720265955567333 std 0.0024721825621897855 MeanAvg 0.9873314014864271
aptos2015 TTA None Mean 0.471382231210149 std 0.02805589901308936 MeanAvg 0.4984512005316677
