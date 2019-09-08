import pytest
from sklearn.metrics import cohen_kappa_score

from retinopathy.dataset import get_datasets
from retinopathy.inference import compute_cdf, run_model_inference_via_dataset, reg_predictions_to_submission, reg_cdf_predictions_to_submission


@pytest.mark.parametrize(['checkpoint_file', 'tta'], ['', None])
def test_cdf_rounding(checkpoint_file, tta):
    train_ds, valid_ds = get_datasets(data_dir='../data', use_aptos2019=True, fold=0)
    cdf = compute_cdf(valid_ds.targets)

    p = run_model_inference_via_dataset(model_checkpoint=checkpoint_file,
                                        dataset=valid_ds,
                                        tta=tta,
                                        batch_size=16,
                                        apply_softmax=False,
                                        workers=6)

    diagnosis = reg_predictions_to_submission(p)['diagnosis'].values
    score = cohen_kappa_score(diagnosis, valid_ds.targets, weights='quadratic')
    print(score)

    cdf_diagnosis = reg_cdf_predictions_to_submission(p, cdf)['diagnosis'].values
    cdf_score = cohen_kappa_score(cdf_diagnosis, valid_ds.targets, weights='quadratic')
    print(cdf_score)
