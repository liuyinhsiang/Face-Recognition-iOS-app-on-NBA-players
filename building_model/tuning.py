# command: python tuning.py --dataset aligned_dataset_52_train --model weights --suffix mobilev2
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from martin_network.martin_vggnet import MartinVGGNet
from martin_network.mobilenetv2 import MartinMobileNetV2
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.applications import MobileNetV2
from keras.applications.mobilenetv2 import preprocess_input
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
    help="path to output model")

ap.add_argument("-s", "--suffix", required=True,
    help="annotation of the model")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 60
FINE_EPOCHS = 5
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (256, 256, 3)
SUFFIX = args["suffix"]
 
# initialize the data and labels
data = []
labels = []
 
# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
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

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
    data.nbytes / (1024 * 1000.0)))
 
# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
 
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
    labels, test_size=0.1, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

# initialize the MobileNetV2
print("[INFO] compiling model...")
model = MartinMobileNetV2.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
    depth=IMAGE_DIMS[2], classes=len(lb.classes_))


# def training(weights_path, layer, opt,SUFFIX,number, aug, trainX, trainY, BS, testX, testY, EPOCHS):
def training(layer, opt, number, epochs, weights=True):
    weights_path = os.path.join(os.path.abspath(args["model"]), 'model_weights-t-'+SUFFIX+'.h5')
    if weights == False:
        pass
    else:
        model.load_weights(weights_path)

    based_model_last_block_layer_number = layer
    for layer in model.layers[:based_model_last_block_layer_number]:
        layer.trainable = False
    for layer in model.layers[based_model_last_block_layer_number:]:
        layer.trainable = True
    model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])
    callbacks_list = [
        ModelCheckpoint(weights_path, monitor='val_acc', verbose=1, save_best_only=True),
        TensorBoard(log_dir='tensorboard/mobilenet-v2-fine-tune-t-'+str(number)+'-'+SUFFIX, 
        histogram_freq=0, write_graph=False, write_images=False)
    ]
    print("[INFO] Fine tuning the model..."+str(number))
    history = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=epochs, verbose=1, callbacks=callbacks_list)
    return history


# opt = 'rmsprop'
opt = SGD(lr=1e-4, momentum=0.9)
H = training(0,opt,0,EPOCHS,False)
# history1 = training(37,opt,1,EPOCHS)
# history2 = training(64,opt,2,EPOCHS)
# history3 = training(91,opt,3,FINE_EPOCHS)
# history4 = training(126,opt,4,FINE_EPOCHS)
# history5 = training(152,opt,5,FINE_EPOCHS)

# save the model to disk
print("[INFO] serializing network...")
model.save("aligned_nba52_MobilenetV2_t_"+SUFFIX+".model")
 
# save the label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open("lb_t_"+SUFFIX+".pickle", "wb")
f.write(pickle.dumps(lb))
f.close()


def ploting(history, name, epoches):
    plt.style.use("ggplot")
    plt.figure()
    N = epoches
    H = history
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="upper left")
    plt.savefig(name+SUFFIX+".png")


# plot the training loss and accuracy
ploting(H,"plot-t-",EPOCHS)
# ploting(history1,"final_plot1_t_",EPOCHS)
# ploting(history2,"final_plot2_t_",FINE_EPOCHS)
# ploting(history3,"final_plot3_t_",FINE_EPOCHS)
# ploting(history4,"final_plot4_t_",FINE_EPOCHS)
# ploting(history5,"final_plot5_t_",FINE_EPOCHS)
