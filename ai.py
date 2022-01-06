from fastai.vision.all import *
import os
import torch


def label(path):
    return path.parent.name


print(torch.cuda.get_device_name(0))
images = get_image_files(os.getcwd())
data = ImageDataLoaders.from_path_func(os.getcwd(), images, label, bs=40, num_workers=0)
learn = cnn_learner(data, models.resnet18, metrics=error_rate)
learn.fine_tune(4, base_lr=1.0e-02)
learn.export()
