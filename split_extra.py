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

train = pd.read_csv("train_extra.csv")

turtle_ids = train["turtle_id"]
turtle_id_set = set()

#classes = list(set(turtle_ids))
for i, turtle_id in enumerate(turtle_ids):
    turtle_id_set.add((turtle_id, i))

classes = list(turtle_id_set)
random.shuffle(classes)
classes = np.array(classes)

#val_len = int(len(classes) / 30)
val_len = int(len(classes) / 5)

prev, nxt = 0, val_len
val_folds = []
train_folds = []

for i in range(5):
    train_folds.append([int(el) for el in np.concatenate([classes[:prev][:, 1], classes[nxt:][:, 1]], 0)])
    val_folds.append([int(el) for el in classes[prev:nxt][:, 1]])
    prev = nxt
    nxt = nxt + val_len

for i in range(5):
    train.loc[val_folds[i][int(0.2 * len(val_folds[i])):]].to_csv(f"extra_folds2/train_folds{i}.csv")
    train.loc[val_folds[i][:int(0.2 * len(val_folds[i]))]].to_csv(f"extra_folds2/val_folds{i}.csv")

    #train.loc[train_folds[i]].to_csv(f"extra_folds/train_fold{i}.csv")
    #train.loc[val_folds[i]].to_csv(f"extra_folds/val_fold{i}.csv")
