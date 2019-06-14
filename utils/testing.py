# command: python testing.py -m aligned_nba52_MobilenetV2_t_mobilev2.model -l lb_t_mobilev2.pickle -d images
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
from tqdm import tqdm
from keras.models import Model
from imutils import paths
import random
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained model")
ap.add_argument("-l", "--labelbin", required=True, help="path to label binarizer")
ap.add_argument("-d", "--dataset", required=True, help="path to test dataset")
args = vars(ap.parse_args())


IMAGE_DIMS = (256, 256, 3)


data = []
labels = []

print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))

for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)
 
    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)


testX, testY = data, labels

model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

# let the model make prediction on the test data
predY = model.predict(testX, verbose=1)

pre = []
count = 0
result = {}

# retrieve predictions and examine with the correct labels to 
# form the result and store it with a dictionary
for py in predY:
    correct_idx = np.argmax(testY[count])
    correct_label = lb.classes_[correct_idx]
    idx = np.argmax(py)
    label = lb.classes_[idx]
    pre.append([idx])
    if correct_label not in result.keys():
        result[correct_label]=[0,0]
    else:
        pass

    if label == correct_label:
        result[correct_label][0] += 1
    else:
        result[correct_label][1] += 1
    count += 1

correct_count = 0
incorrect_count = 0
total_count = 0

# organize the result and compute the accuracy
for (x,y) in result.items():
    correct_count += y[0]
    incorrect_count += y[1]
    total_count = correct_count + incorrect_count

accuracy = correct_count/total_count
print("Correct Count: {}\nIncorrect Count: {}\nTotal Count: {}\nAccuracy: {:.1%}"
    .format(correct_count, incorrect_count, total_count, accuracy))


