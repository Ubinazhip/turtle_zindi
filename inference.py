import torch
import torch.nn as nn
import torch.optim as optim
import timm
from timm.models.efficientnet import *
from timm.models.gluon_resnet import *
from timm.models.swin_transformer import *
from timm.models.densenet import *
import os
import argparse
from torch.utils.data.dataset import Dataset
import os
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default="v2s")
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--model_path', type=str, default="/home/turtle_recall/v2s_fold0.pth")

args = parser.parse_args()

if args.model_type == "b7":
    model = tf_efficientnet_b7(pretrained=False, drop_rate=0.3, drop_path_rate=0.2)
    model.classifier = nn.Linear(2560, args.num_classes)
if args.model_type == "b8":
    model = tf_efficientnet_b8(pretrained=True)
    model.classifier = nn.Linear(2816, args.num_classes)
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
elif args.model_type == "v2l":
    model = tf_efficientnetv2_l_in21ft1k(pretrained=False, drop_rate=0.3, drop_path_rate=0.2)
    model.classifier = nn.Linear(1280, args.num_classes)
elif args.model_type == "v2s":
    model = tf_efficientnetv2_s_in21ft1k(pretrained=False, drop_rate=0.3, drop_path_rate=0.2)
    model.classifier = nn.Linear(1280, args.num_classes)
elif args.model_type == "v2m":
    model = tf_efficientnetv2_m_in21ft1k(pretrained=False, drop_rate=0.3, drop_path_rate=0.2)
    model.classifier = nn.Linear(1280, args.num_classes)
elif args.model_type == "swin":
    model = swin_base_patch4_window12_384(pretrained=False)
    model.head = nn.Linear(1024, args.num_classes)
elif args.model_type == "densenet":
    model = densenet121(pretrained=True)
    model.classifier = nn.Linear(1024, args.num_classes)
elif args.model_type == "seres":
    model = gluon_seresnext50_32x4d(pretrained=True)
    model.fc = nn.Linear(2048, args.num_classes)
else:
    print("No such model in our zoo!")
    exit()

im_size = 680
crop_size = 640

def data_transforms():
    return {
        'test': albumentations.Compose([
            albumentations.Resize(im_size, im_size),
            albumentations.CenterCrop(crop_size, crop_size),
            albumentations.Normalize(),
            ToTensorV2(),
        ])
}

class CustomDataset(Dataset):
    def __init__(self, transform, split):
        self.transform = transform
        self.split = split
        self.data_folder = "/home/tf/train_data"
        self.csv_folder = "/home/tf/turtle_data"

        self.data = pd.read_csv(os.path.join(self.csv_folder, f"{self.split}.csv"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.data_folder, self.data["image_id"][idx] + ".JPG"))
        transformed = self.transform(image=image)
        image = transformed["image"]

        return image

transforms = data_transforms()

model = torch.nn.DataParallel(model)
model = model.cuda()
#model.load_state_dict(torch.load("/home/turtle_recall/v2s_fold0_no_sampling.pth"))
model.load_state_dict(torch.load(args.model_path))
image_dataset = CustomDataset(transforms["test"], "test")

dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

df = pd.read_csv("/home/tf/turtle_data/train.csv")
class_id_map = {}

for i in range(len(df)):
    class_id_map[df["class"][i]] = df["turtle_id"][i]

def test_model(model, dataloader):
    model.eval()
    predicted = torch.tensor([]).cuda()

    for inputs in tqdm(dataloader):
        inputs = inputs.cuda()

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            top5_outs = torch.topk(outputs, 5)[1]
            predicted = torch.cat((predicted, top5_outs), 0)

    return np.array(predicted.cpu())

outputs = test_model(model, dataloader)
submission = {"image_id": [], "prediction1": [], "prediction2": [], "prediction3": [], "prediction4": [], "prediction5": []}

count = 0

for i in range(len(outputs)):
    submission["image_id"].append(image_dataset.data["image_id"][i])

    if outputs[i][0] == 100 or outputs[i][1] == 100 or outputs[i][2] == 100 or outputs[i][3] == 100 or outputs[i][4] == 100:
        count += 1

    if outputs[i][0] != 100:
        submission["prediction1"].append(class_id_map[int(outputs[i][0])])
    else:
        submission["prediction1"].append("new_turtle")

    if outputs[i][1] != 100:
        submission["prediction2"].append(class_id_map[int(outputs[i][1])])
    else:
        submission["prediction2"].append("new_turtle")

    if outputs[i][2] != 100:
        submission["prediction3"].append(class_id_map[int(outputs[i][2])])
    else:
        submission["prediction3"].append("new_turtle")

    if outputs[i][3] != 100:
        submission["prediction4"].append(class_id_map[int(outputs[i][3])])
    else:
        submission["prediction4"].append("new_turtle")

    if outputs[i][4] != 100:
        submission["prediction5"].append(class_id_map[int(outputs[i][4])])
    else:
        submission["prediction5"].append("new_turtle")

pd.DataFrame(submission).to_csv("submission.csv", index=False)
print(count)
