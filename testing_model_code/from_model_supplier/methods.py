import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np
#import torch
import time

# set seed for repetability
np.random.seed(seed=42)

# set device for torch
'''
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("")
print(f'Using {device} for inference')
print("")
'''

# load Resnet
def get_ResNet50(mode):
    if mode:    
        model = ResNet50(weights=None)
    else:
        model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=False)
        model = model.to(device) 
    return model

# function that creates random images
def rand_image_TF(image_size, series_size):
    if series_size > 0:
        image = np.random.randint(255, size=(series_size,image_size[0],image_size[1],3), dtype=np.uint8)
    else:
        image = np.random.randint(255, size=(image_size[0],image_size[1],3), dtype=np.uint8)
    return image

# create rand image torch tensor
def rand_image_PT(image_size, series_size):
    if series_size > 0:
        image = torch.randint(0,255,(series_size,3,image_size[0],image_size[0]), dtype=torch.float)
    else:
        image = torch.randint(0,255,(3,image_size[0],image_size[0]), dtype=torch.float)
    return image

def infere_TF(model,image_size,series_size,mode,res=None):
    start = time.time()
    if mode:
        image = rand_image_TF(image_size,series_size)
        model.predict(image)
    else:
        image = rand_image_PT(image_size,series_size)
        image = image.to(device)
        model(image)
    end = time.time()
    # display elapsed time for the inference
    elaps_time = (end-start)/image.shape[0]
    # store data
    if res is not None:
        res.append(elaps_time)
    print(f"elapsed time per image: {elaps_time}")


# print stats on bemchmark

def print_stat(res,name):
    avgerage = np.mean(res)
    minimum  = np.min(res)
    maximum  = np.max(res)
    standdev = np.std(res)/avgerage
    print(f"{name}, inference time [s]: mean = {int(avgerage*1000)/1000:.3f}, std % = {standdev*100:2.1f}, max = {int(maximum*1000)/1000:.3f}, min = {int(minimum*1000)/1000:.3f}")
 

    