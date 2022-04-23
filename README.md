# turtle_zindi
8th place solution for ["Turtle Recall: Conservation Challenge"](https://zindi.africa/competitions/turtle-recall-conservation-challenge)

# Task
Classificaiton of sea turtle based on their facial scales, which are considered to be unique. For each given image, the model should output turtle's unique id. 
The test set might have a turtles that train set doesn't have, in that case the model should output 'new_turtle'. There are 100 unique turtles.

## Evaluation 
Top 5 mean average precision (mAP)

# Train
- We have divided the dataset into 5 folds. We included the extra images that did not belong to the 100 turtles as the "new turtle" class. 
- Since the 'new turtle' class caused class imbalance, we used WeightedRandomSampler
- For augmentations, we have used [albumentation](https://albumentations.ai/). 
```python
transforms = albumentations.Compose([
            albumentations.Resize(680, 680),
            albumentations.RandomCrop(640, 640),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ShiftScaleRotate(rotate_limit=15, scale_limit=0.15, shift_limit=0.1, p=0.9),
            albumentations.OneOf([
                albumentations.Blur(blur_limit=7, p=1.0),
                albumentations.MotionBlur(),
                albumentations.GaussNoise(),
                albumentations.ImageCompression(quality_lower=75)
            ], p=0.5),
            albumentations.Cutout(num_holes=12, max_h_size=35, max_w_size=35, p=0.5),
            albumentations.Normalize(),
            ToTensorV2(),
        ]),
```

- We trained 5 efficientNet models from [timm](https://github.com/rwightman/pytorch-image-models): EfficientNet b3, b4, b5, b6, b7.
- 12 best models in terms of top5 mAP in validation and public test set had been used for ensembling, namely:
  * EfficientNet b6 trained on fold 3 and 4
  * EfficientNet b3 trained on fold 0, 1, 2
  * EfficientNet b7 trained on fold 0
  * EfficientNet b5 trained on fold 0, 2, 4
  * EfficientNet b4 trained on fold 2, 3, 4   
- In order to train the model use [train.py](https://github.com/Ubinazhip/turtle_zindi/blob/master/train.py) file. The best model will be saved in the current directory. For example, if you want to train EfficientNet b3 in fold 3,
```python
CUDA_VISIBLE_DEVICS=0 python3 train.py --model_type b3 --batch_size 4 --epochs 50 --fold 0
```
- For ensembling the models use [ensemble.py](https://github.com/Ubinazhip/turtle_zindi/blob/master/ensemble.py)


# Authors: 
- Aslan Ubingazhibov - aslan.ubingazhibov@alumni.nu.edu.kz
- Aidyn Ubingazhibov - aidyn.ubingazhibov@nu.edu.kz


