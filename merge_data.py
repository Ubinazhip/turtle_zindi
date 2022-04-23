import pandas as pd

main_folds = []
extra_folds = []

for i in range(5):
    main_folds.append(pd.read_csv(f"main_folds/train_folds{i}.csv"))
    extra_folds.append(pd.read_csv(f"extra_folds2/train_folds{i}.csv"))

combined = []

for i in range(5):
    main_folds[i] = main_folds[i].append(extra_folds[i])
    combined.append(main_folds[i])

for i in range(5):
    combined[i].to_csv(f"folds2/train_folds{i}.csv", index=False)

main_folds_val = []
extra_folds_val = []

for i in range(5):
    main_folds_val.append(pd.read_csv(f"main_folds/val_folds{i}.csv"))
    extra_folds_val.append(pd.read_csv(f"extra_folds2/val_folds{i}.csv"))

combined_val = []

for i in range(5):
    main_folds_val[i] = main_folds_val[i].append(extra_folds_val[i])
    combined_val.append(main_folds_val[i])

for i in range(5):
    combined_val[i].to_csv(f"folds2/val_folds{i}.csv", index=False)
