import argparse

import torch


def clean_checkpoint(src_fname, dst_fname):
    checkpoint = torch.load(src_fname)

    keys = [
        'criterion_state_dict',
        'optimizer_state_dict',
        'scheduler_state_dict',
    ]

    for key in keys:
        if key in checkpoint:
            del checkpoint[key]

    torch.save(checkpoint, dst_fname)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='+')
    args = parser.parse_args()
    for c in args.input:
        clean_checkpoint(c, c)


if __name__ == '__main__':
    main()
