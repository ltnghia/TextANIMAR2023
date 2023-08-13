import pandas as pd 
import numpy as np
import random


df = pd.read_csv('./content/original/mapped_TextQuery_multiview_rotated_GT_train.csv')
train = df.loc[0:607, :]
test = df.loc[608:, :]
test.index = range(len(test))

train.to_csv("./content/mapped_TextQuery_GT_Train.csv", index = False)
test.to_csv('./content/mapped_TextQuery_GT_Val.csv', index = False)
df = pd.read_csv('./content/original/mapped_TextQuery_Train.csv')

train = df.loc[0:78, :]
for i in train.index:
  S = train.iloc[i][1]
  S = S.split(" is ")[0]
  train.iloc[i][1] = S

test = df.loc[79:, :]
test.index = range(len(test))
for i in test.index:
    S = test.iloc[i][1]
    S = S.split(" is ")[0]
    test.iloc[i][1] = S

train.to_csv("./content/mapped_TextQuery_Train.csv", index = False)
test.to_csv('./content/mapped_TextQuery_Val.csv', index = False)