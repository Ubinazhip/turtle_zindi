import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import random
import os

def seed_everything(seed=1234):
   random.seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   np.random.seed(seed)


seed_everything(1234)

train = pd.read_csv("train.csv")
classes = train["class"]
location = train["image_location"]

X = list(train["image_id"])
location_map = {"Top": 0, "top": 0, "Left": 1, "left": 1, "Right": 2, "right": 2}

y = [[cl, location_map[location]] for cl, location in zip(classes, location)]

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=False, random_state=0)

fold = 0

for train_index, test_index in mskf.split(X, y):
   train.loc[train_index].to_csv(f"main_folds/train_folds{fold}.csv", index=False)
   train.loc[test_index].to_csv(f"main_folds/val_folds{fold}.csv", index=False)
   fold += 1


