import torch
import torch.nn as nn
import torch.optim as optim
from dataset import CustomDataset, data_transforms
import argparse
import timm
from timm.models.efficientnet import *
from timm.models.swin_transformer import *
from timm.models.gluon_resnet import *
from timm.models.densenet import *
from timm.models.senet import legacy_senet154
from timm.models.tresnet import tresnet_l
from timm.models.resnet import resnext101_32x4d
from timm.models.xception_aligned import xception71
import os
from runner import train_model
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random


parser = argparse.ArgumentParser()

parser.add_argument('--model_type', type=str, default="v2s")
parser.add_argument('--folder', type=str, default="folds")
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--num_classes', type=int, default=101)
parser.add_argument('--patience', type=int, default=8)
parser.add_argument('--L2', type=float, default=5e-5)
parser.add_argument('--sampling', type=str, default="sampler")

args = parser.parse_args()

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(1234)

if args.model_type == "b8":
    model = tf_efficientnet_b8(pretrained=True)
    model.classifier = nn.Linear(2816, args.num_classes)
elif args.model_type == "b7":
    model = tf_efficientnet_b7(pretrained=True, drop_rate=0.3, drop_path_rate=0.2)
    model.classifier = nn.Linear(2560, args.num_classes)
elif args.model_type == "b6":
    model = tf_efficientnet_b6_ns(pretrained=True)
    model.classifier = nn.Linear(2304, args.num_classes)
elif args.model_type == "b5":
    model = tf_efficientnet_b5_ns(pretrained=True)
    model.classifier = nn.Linear(2048, args.num_classes)
elif args.model_type == "b4":
    model = tf_efficientnet_b4_ns(pretrained=True)
    model.classifier = nn.Linear(1792, args.num_classes)
elif args.model_type == "b3":
    model = tf_efficientnet_b3_ns(pretrained=True)
    model.classifier = nn.Linear(1536, args.num_classes)
elif args.model_type == "b2":
    model = tf_efficientnet_b2_ns(pretrained=True)
    model.classifier = nn.Linear(1408, args.num_classes)
elif args.model_type == "l2":
    model = tf_efficientnet_l2_ns(pretrained=True, drop_rate=0.5)
    model.classifier = nn.Linear(5504, args.num_classes)
elif args.model_type == "v2l":
    model = tf_efficientnetv2_l_in21ft1k(pretrained=True, drop_rate=0.3, drop_path_rate=0.2)
    model.classifier = nn.Linear(1280, args.num_classes)
elif args.model_type == "v2s":
    model = tf_efficientnetv2_s_in21ft1k(pretrained=True, drop_rate=0.3, drop_path_rate=0.2)
    model.classifier = nn.Linear(1280, args.num_classes)
elif args.model_type == "v2m":
    model = tf_efficientnetv2_m_in21ft1k(pretrained=True, drop_rate=0.3, drop_path_rate=0.2)
    model.classifier = nn.Linear(1280, args.num_classes)
elif args.model_type == "swin":
    model = swin_base_patch4_window12_384(pretrained=True)
    model.head = nn.Linear(1024, args.num_classes)
elif args.model_type == "seres":
    model = gluon_seresnext50_32x4d(pretrained=True)
    model.fc = nn.Linear(2048, args.num_classes)
elif args.model_type == "densenet":
    model = densenet121(pretrained=True)
    model.classifier = nn.Linear(1024, args.num_classes)
elif args.model_type == "tresnet_l":
    model = tresnet_l(pretrained=True, num_classes=101)
elif args.model_type == "resnext":
    model = resnext101_32x4d(pretrained=True, num_classes=101)
elif args.model_type == "xception":
    model = xception71(pretrained=True, num_classes=101)
elif args.model_type == "senet":
    model = legacy_senet154(pretrained=True, num_classes=101)
else:
    print("No such model in our zoo!")
    exit()

transform = data_transforms

if args.model_type == "swin" or args.model_type == "l2":
    from dataset import data_transforms2
    transform = data_transforms2

model = torch.nn.DataParallel(model)
model = model.cuda()
transforms = transform()

lr = 3e-4

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.L2)
image_datasets = {x: CustomDataset(transforms[x], x, args.folder, args.fold) for x in ['train', 'val']}

#sampler
classes = np.array(image_datasets["train"].data["class"])
class_sample_count = np.array([(classes == i).sum() for i in range(args.num_classes)])
weights = 1. / class_sample_count

samples_weight = np.array([weights[t] for t in classes])
samples_weight = torch.from_numpy(samples_weight)
samples_weight = samples_weight.double()

#criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_sample_count).cuda().float())
if args.sampling == "weighted":
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).cuda().float())
else:
    criterion = nn.CrossEntropyLoss()

sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))

if args.sampling == "sampler":
    dataloaders_dict = {"train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=args.batch_size, sampler=sampler, shuffle=False, num_workers=4),
                        "val": torch.utils.data.DataLoader(image_datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=4)}
else:
    dataloaders_dict = {
        "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=args.batch_size,
                                             shuffle=True, num_workers=4),
        "val": torch.utils.data.DataLoader(image_datasets["val"], batch_size=args.batch_size, shuffle=False,
                                           num_workers=4)}

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, factor=0.1)
best_model_wts = train_model(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs=args.epochs)
torch.save(best_model_wts, f"{args.model_type}_fold{args.fold}_{args.sampling}.pth")
