import argparse
import json

import pandas as pd

from pytorch_toolbelt.utils.fs import auto_file, change_extension


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--submission', type=str)
    args = parser.parse_args()
    submission = auto_file(args.submission)
    submission_kaggle = change_extension(submission, '_kaggle.py')
    submission_df = pd.read_csv(submission)

    json_data = json.loads(submission_df.to_json())
    with open(submission_kaggle, 'w') as f:
        f.write('import pandas as pd\n')
        f.write('submission = ')
        json.dump(json_data, f)
        # f.write(json_data)
        f.write('\n')
        f.write('submission = pd.DataFrame.from_dict(submission)\n')
        f.write('submission.to_csv(\'submission.csv\', index=None)\n')


if __name__ == '__main__':
    main()
