import subprocess
import os
import argparse
from tqdm import tqdm

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=True,
	help=".txt file path which contains of name list")
args = vars(ap.parse_args())

path = args['file']

with open(path) as f:

    for readline in f:
        querry = readline.strip("\n")
        print(querry)
        folder_name = querry.replace(" ", "_")
        print('dataset/'+folder_name)
        if not os.path.exists('dataset/'+folder_name):
            os.mkdir('dataset/'+folder_name)

        command = "python3 search_bing_api.py --query "+"'"+querry+"'"+ " --output dataset/"+folder_name

        os.system(command)


os.system('python3 folder_naming.py -f dataset')




        
        
