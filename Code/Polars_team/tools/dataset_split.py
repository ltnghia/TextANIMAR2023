import pandas as pd
from sklearn.model_selection import train_test_split

SOURCE_FILE = "dataset/csv/TextQuery_GT_Train.csv"
TEST_RATIO = 0.1

# read the CSV file into a DataFrame
df = pd.read_csv(SOURCE_FILE, delimiter=';')

# shuffle the DataFrame
df = df.sample(frac=1, random_state=42)

# group the DataFrame by "Model ID"
groups = df.groupby('Model ID')

# split the groups into train and test sets
train_ids, test_ids = train_test_split(list(groups.groups.keys()), test_size=TEST_RATIO, random_state=42)

# create separate DataFrames for train and test
train_df = pd.concat([groups.get_group(model_id) for model_id in train_ids])
test_df = pd.concat([groups.get_group(model_id) for model_id in test_ids])

# save the train and test DataFrames to CSV files
train_df.to_csv('TextQuery_GT_Train_split.csv', index=False, sep=";")
test_df.to_csv('TextQuery_GT_Val_split.csv', index=False, sep=";")
