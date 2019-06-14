#***************************************************************************************
#*  Title: Face Alignment with OpenCV and Python
#*  Author: Adrian Rosebrock
#*  Date: May 22, 2017
#*  Code version: 1.0
#*  Availability: https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
#***************************************************************************************
# modified from source

# command: $ python multi_align_faces.py -p shape_predictor_68_face_landmarks.dat -i images -o unknown

# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import os
from tqdm import tqdm
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
help="path to facial landmark predictor")
ap.add_argument("-i", "--input", required=True,
help="path to input folder")
ap.add_argument("-o", "--output", required=True,
help="path to output folder")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fa = FaceAligner(predictor, desiredFaceWidth=224)

input_path = args['input']
folder_list = os.listdir(input_path)

output_path = args['output']


def image_align(input_path, output_path):
    image_paths = os.listdir(input_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    for name in tqdm(image_paths, total=len(image_paths), unit='image'):
        try:
            image_path = os.path.join(input_path, name)
            new_image_path = os.path.join(output_path, name)

            # load the input image, resize it, and convert it to grayscale
            image = cv2.imread(image_path)
            
            image = imutils.resize(image, width=800)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 2)

            # loop over the face detections
            for rect in rects:
                # extract the ROI of the *original* face, then align the face
                # using facial landmarks
                (x, y, w, h) = rect_to_bb(rect)
                faceAligned = fa.align(image, gray, rect)
            

                cv2.imwrite(new_image_path, faceAligned, [cv2.IMWRITE_JPEG_QUALITY, 100])
        except:
            print(name)


if not os.path.exists(output_path):
    os.mkdir(output_path)

for folder in tqdm(folder_list, total=len(folder_list), unit='folder'):
    print('\n')
    print(folder)
    output_folder = os.path.join(output_path, folder)
    folder_name = os.path.join(input_path, folder)
    
    if os.path.isdir(folder_name):
        image_align(folder_name, output_folder)
    else:
        pass




