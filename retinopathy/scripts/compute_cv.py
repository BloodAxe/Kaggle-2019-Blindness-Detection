import argparse
from collections import defaultdict

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+')
    args = parser.parse_args()

    metrics = defaultdict(list)
    names = {
        'cls/kappa',
        'reg/kappa',
        'ord/kappa',
        'ord/f1_macro',
        'ord/f2_micro',
        'ord/accuracy',
    }

    checkpoints = args.input
    for checkpoint in checkpoints:
        checkpoint = torch.load(checkpoint, map_location='cpu')
        for name in names:
            metrics[name].append(checkpoint['valid_metrics'].get(name, 0))

    for name in names:
        values = metrics[name]
        print('{:10.10s}'.format(name), '{:.4f}'.format(np.mean(values)), '{:.4f}'.format(np.std(values)), values)


if __name__ == '__main__':
    main()
