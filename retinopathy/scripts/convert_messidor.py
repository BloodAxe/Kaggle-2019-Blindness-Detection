import csv

import pandas as pd
import xlrd
from pytorch_toolbelt.utils.fs import change_extension, id_from_fname


def csv_from_excel(fname, sheet_name='Feuil1') -> pd.DataFrame:
    wb = xlrd.open_workbook(fname)
    sh = wb.sheet_by_name(sheet_name)
    csv_file = change_extension(fname, '.csv')
    your_csv_file = open(csv_file, 'w', encoding='utf-8')
    wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)

    for rownum in range(sh.nrows):
        wr.writerow(sh.row_values(rownum))

    your_csv_file.close()
    return pd.read_csv(csv_file)


files = [
    'Annotation_Base11.xls',
    'Annotation_Base12.xls',
    'Annotation_Base13.xls',
    'Annotation_Base14.xls',
    'Annotation_Base21.xls',
    'Annotation_Base22.xls',
    'Annotation_Base23.xls',
    'Annotation_Base24.xls',
    'Annotation_Base31.xls',
    'Annotation_Base32.xls',
    'Annotation_Base33.xls',
    'Annotation_Base34.xls',
]

df = pd.DataFrame()
for file in files:
    df = pd.concat((df, csv_from_excel(file)))

df = df.rename(columns={"Retinopathy grade": "diagnosis",
                        "Image name": "image_fname"})

df = df.drop(columns=['Ophthalmologic department',
                      'Risk of macular edema '])
df['idcode'] = df['image_fname'].apply(id_from_fname)
df['diagnosis'] = df['diagnosis'].apply(int)

print(df.head())
df.to_csv('train.csv', index=None)
