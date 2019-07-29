import argparse

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+')
    args = parser.parse_args()

    cv = []
    checkpoints = args.input
    for checkpoint in checkpoints:
        checkpoint = torch.load(checkpoint)
        cv.append(checkpoint['valid_metrics']['kappa_score'])

    print(np.mean(cv))
    print(np.std(cv))


if __name__ == '__main__':
    main()
