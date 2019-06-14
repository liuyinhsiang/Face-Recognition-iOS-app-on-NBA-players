#***************************************************************************************
#*  Title: Face Alignment with OpenCV and Python
#*  Author: Adrian Rosebrock
#*  Date: May 22, 2017
#*  Code version: 1.0
#*  Availability: https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
#***************************************************************************************
# modified from source

# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=256)

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
new_name = args['image'][:-4]+'_alligned.jpg'
image = imutils.resize(image, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 

rects = detector(gray, 2)

# loop over the face detections
for rect in rects:
    # extract the ROI of the *original* face, then align the face
    # using facial landmarks
    (x, y, w, h) = rect_to_bb(rect)
    faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
    faceAligned = fa.align(image, gray, rect)
 

    cv2.imwrite(new_name, faceAligned)



