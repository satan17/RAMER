import os
from unittest.mock import inplace

import pandas as pd
from sklearn.model_selection import train_test_split

base_dir = os.path.join('datasets', 'affectNet')
df = pd.read_csv(os.path.join(base_dir, 'labels.csv'))
# print(len(df))

# drop unwanted emotions, keep only common emotions
# df = df[~df['label'].isin(['contempt', 'surprise'])]
df.drop(df[df['label'] == 'contempt'].index, inplace=True)
df.drop(df[df['label'] == 'surprise'].index, inplace=True)
df.drop(df[df['label'] == 'neutral'].index, inplace=True)
df.drop(df[df['pth'].str.startswith('surprise')].index, inplace=True)
df.drop(df[df['pth'].str.startswith('contempt')].index, inplace=True)
df.drop(df[df['pth'].str.startswith('neutral')].index, inplace=True)


for index, row in df.iterrows():
    path = row['pth']
    file_path = os.path.join('datasets', 'affectNet', 'images', path.split("/")[0], path.split("/")[1])
    if not os.path.isfile(file_path):
        df = df.drop(index=index)
    df.loc[index, 'pth'] = file_path

balanced_df = (
    df.groupby("label", group_keys=False)
      .apply(lambda x: x.sample(n=df["label"].value_counts().min(),
                                random_state=42))
)

print(df.groupby("label").size())
print("--------------------------------")
print(balanced_df.groupby("label").size())

# print(balanced_df.head(10))

# split
train_df, test_df = train_test_split(balanced_df, test_size=0.1, stratify=balanced_df['label'], random_state=82)

# add split column
train_df = train_df.copy()
test_df = test_df.copy()
train_df["split"] = "train"
test_df["split"] = "val"

print(train_df['label'].value_counts())
print("-------------------------------")
print(test_df['label'].value_counts())

# combine back
df_with_split = pd.concat([train_df, test_df]).sort_index()
df_with_split.drop(columns=['Unnamed: 0', 'relFCs'], inplace=True)
# rename column first
df_with_split = df_with_split.rename(columns={'pth': 'filepath'})
# print(df_with_split.head(10))
# df_with_split.to_csv(os.path.join(base_dir, 'updated_labels.csv'), index=False)