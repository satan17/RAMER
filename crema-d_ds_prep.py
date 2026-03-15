import os

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

mapping = {'ANG':'anger',
           'DIS':'disgust',
           'FEA':'fear',
           'HAP':'happy',
           'SAD':'sad'}

video_dir = os.path.join('datasets', 'crema-d', 'video')

df = pd.DataFrame(columns=['filepath', 'label', 'speaker', 'gender'])

male = [1001, 1005, 1011, 1014, 1015, 1016, 1017, 1019, 1022, 1023, 1026, 1027, 1031, 1032, 1033, 1034, 1035, 1036, 1038, 1039, 1040, 1041, 1042, 1044, 1045, 1048, 1050, 1051, 1057, 1059, 1062, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1077, 1080, 1081, 1083, 1085, 1086, 1087, 1088, 1090]
# female = [1002, 1003, 1004, 1006, 1007, 1008, 1009, 1010, 1012, 1013, 1018, 1020, 1021, 1024, 1025, 1028, 1029, 1030, 1037, 1043, 1046, 1047, 1049, 1052, 1053, 1054, 1055, 1056, 1058, 1060, 1061, 1063, 1072, 1073, 1074, 1075, 1076, 1078, 1079, 1082, 1084, 1089, 1091]

for file in os.listdir(video_dir):
    emo = file.split("_")[2]
    speaker = file.split("_")[0]
    gender = "male" if int(speaker) in male else "female"
    if emo not in mapping:
        continue
    df.loc[len(df)] = [os.path.join(video_dir, file), mapping[emo], speaker, gender]

print(df.groupby("gender").size())
print(df.head(5))

df = df.reset_index(drop=True)

# create stratification label: label + gender
stratify_labels = df["label"].astype(str) + "_" + df["gender"].astype(str)

X = df.index.values
y = stratify_labels.values
groups = df["speaker"].values

sgkf = StratifiedGroupKFold(
    n_splits=5,      # ~80/20 split
    shuffle=True,
    random_state=42
)

train_idx, test_idx = next(sgkf.split(X, y, groups))

train_df = df.iloc[train_idx].copy()
test_df  = df.iloc[test_idx].copy()

# add split column
train_df["split"] = "train"
test_df["split"] = "val"

# Mandatory sanity checks (do not skip)

# 1. No speaker leakage
assert set(train_df.speaker).isdisjoint(set(test_df.speaker))

# 2. Label distribution
print(train_df.label.value_counts())
print(test_df.label.value_counts())

# 3. Gender distribution
print(train_df.gender.value_counts())
print(test_df.gender.value_counts())

df_with_split = (
    pd.concat([train_df, test_df])
      .sort_index()
)

# df_with_split.to_csv(os.path.join('datasets', 'crema-d', "updated_labels.csv"), index=False)