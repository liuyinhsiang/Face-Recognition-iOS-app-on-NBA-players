import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True,
help="Changing naming in folder: ")

args = vars(ap.parse_args())


def batch_rename(path):
    for fname in os.listdir(path):
        new_fname = fname[:8] + '.jpg'
        print( os.path.join(path, fname))
        os.rename(os.path.join(path, fname), os.path.join(path, new_fname))

path = args["folder"]

for f in os.listdir(path):
    # print(f)
    pathname = os.path.join(path, f)
    if os.path.isdir(pathname):
        # pathname = os.path.join(path, f)
        # print("cool")
        batch_rename(pathname)
    else:
        pass


