import os
import matplotlib.pyplot as plt
import random
from numpy import asarray
from sklearn import svm
from sklearn.model_selection import train_test_split

class Image:
    def __init__(self, image, target):
        self.image = image
        self.target = target
 
# get the path/directory
images = []
folder_dir_hot_dog = "images/train/hotdog"
for hot_dog_image in os.listdir(folder_dir_hot_dog):
    images.append(Image((plt.imread(f"{folder_dir_hot_dog}/{hot_dog_image}")),"hot_dog"))



folder_dir_not_hot_dog = "images/train/nothotdog"
for not_hot_dog_image in os.listdir(folder_dir_not_hot_dog):
    images.append(Image((plt.imread(f"{folder_dir_not_hot_dog}/{not_hot_dog_image}")),"not_hot_dog"))

random.shuffle(images)

images_without_target = []
target_without_images = []

for image in images:
    images_without_target.append(image.image)
    target_without_images.append(image.target)













