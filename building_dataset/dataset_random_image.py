### command:
### $ python random_image.py -i imagefolder -o outputfolder -q 300

import os, random, shutil
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
        help="path of input directory of images")
ap.add_argument("-o", "--output", required=True,
        help="path to output directory of images")
ap.add_argument("-q", "--quantity", required=True,
        help="selecting how many images")
args = vars(ap.parse_args())

input_path = args['input']
output_path = args['output']
quantity = args['quantity']

def random_select(input_path, output_path, quantity):
    file_paths = os.listdir(input_path) 
    filenumber=len(file_paths)
    if not os.path.exists(output_path):
            os.mkdir(output_path)
    
    if filenumber > int(quantity):
        print("all good")
        sample = random.sample(file_paths, int(quantity))
        for name in sample:
            shutil.copyfile(input_path+'/'+name, output_path+'/'+name)
    else:
        print("Asked quantity exceed the existing amount")

    return

if not os.path.exists(output_path):
    os.mkdir(output_path)

for folder in os.listdir(input_path):
    new_folder = os.path.join(output_path, folder)
    path_name = os.path.join(input_path, folder)
    if os.path.isdir(path_name):
        random_select(path_name, new_folder, quantity)
    else:
        pass
