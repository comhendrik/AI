

import matplotlib.pyplot as plt
from numpy import asarray

from sklearn import datasets, svm
from sklearn.model_selection import train_test_split



digits = datasets.load_digits()

#flatten images
data = digits.images.reshape((len(digits.images), -1))


#classification
clf = svm.SVC(gamma=0.001)

X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.2, shuffle=False)

clf.fit(X_train, y_train)


#check wether you want to predict your own image or to predict an image of the 
use_own_image = True
if use_own_image:

    #convert image shape (8,8,4) to shape (1,8,8)
    image = plt.imread('1.png')
    print(image.shape)
    new_image = []
    for x in image:
        pixels = []
        for y in x:
            pixels.append(255 - y[0])
        new_image.append(pixels)

    new_image = asarray([new_image])

    #flatten the image

    new_image_data = new_image.reshape((len(new_image),-1))

    plt.title(f"predicted number: {clf.predict(new_image_data)[0]}")

    plt.imshow(new_image[0])
else:
    number = 5

    plt.title(f"predicted number: {clf.predict(data)[number]}, actual number: {digits.target[number]}")

    plt.imshow(digits.images[number])

plt.show()




