#CottonZ 2021.02.03
#Edited from https://docs.opencv.org/master/d1/dc5/tutorial_background_subtraction.html

import cv2 as cv
import argparse
import matplotlib.pyplot as plt
import os
import re
from shutil import rmtree


parser = argparse.ArgumentParser(description='Background subtraction by OpenCV')
parser.add_argument('--image_folder', '-f', dest='image_folder', help='the absolute path of the cropped image directory')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.The original input by this method used in the OpenCV website',  required = False)
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')

args = parser.parse_args()
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

#Input image sequence
if args.image_folder:
    folder_path = args.image_folder.rstrip('/').rstrip('\\')
    bgsub_folder = folder_path+'/bg_subtraction'
    if os.path.isdir(bgsub_folder):
        val = input("Directory exist, Overwrite? [Y]/[N]")
        if val.lower() == "y" or val.lower() == "yes":
            rmtree(bgsub_folder)
            os.mkdir(bgsub_folder)
        elif val.lower() == "n" or val.lower() == "no":
            print("Exiting..........")
            exit(0)
        else:
            print("Input should be yes or no")
            print("Exiting..........")
            exit(0)
    else:
        os.mkdir(bgsub_folder)
    
    #Get file name pattern
    filename = ""
    image_name_list = []
    for file in os.listdir(folder_path):
        if  ".tif" in file:
            image_name_list.append(file)
    if len(image_name_list)==0:
        print("There is no image file in this directory "+folder_path)
        exit(0)
    else:
        filename = image_name_list[0]
        if 'c1t'in filename and 'xy1' in filename and "xy1c1" not in filename:
            NamePattern = dict()
            filename = re.split("c1t|xy1", filename)
            NamePattern[0] =  filename[0]+'c1t'
            NamePattern[1] = 'xy1'+filename[-1]
            NamePattern[2] = len(filename[1])
        elif "c1t" in filename and 'xy1' not in filename: 
            NamePattern = dict()
            filename = re.split("c1t|_cropped", filename) 
            NamePattern[0] =  filename[0]+'c1t'
            NamePattern[1] = '_cropped'+filename[-1]
            NamePattern[2] = len(filename[1])
        elif "xy1c1" in filename: #0--10x--flic(mg1655)--200-1034t0001xy1c1_cropped
            experiment_index = folder_path.split('/')[-1].split('\\')[-1].replace("-_Channel_only", '').rstrip('-').split('-')[-1]
            NamePattern = dict()
            repattern = experiment_index+'t|xy1c1'
            filename = re.split(repattern, filename)
            NamePattern[0] =  filename[0]+experiment_index+'t'
            NamePattern[1] = 'xy1c1'+filename[-1]
            NamePattern[2] = len(filename[1])
        elif "c1" in filename and "c1t" not in filename and "xy1" not in filename: #0--10x-mg1655--200-1048t0001c1_cropped.tif
            experiment_index = folder_path.split('/')[-1].split('\\')[-1].replace("_Channel_only", '').rstrip('-').split('-')[-1]
            print(filename)
            print(experiment_index)
            NamePattern = dict()
            repattern = experiment_index+'t|c1_cropped'
            filename = re.split(repattern, filename)
            NamePattern[0] =  filename[0]+experiment_index+'t'
            NamePattern[1] = 'c1_cropped'+filename[-1]
            NamePattern[2] = len(filename[1])
        elif "c1xy1t" in filename :#0--10x--flic(mg1655)--200-1033c1xy1t0002.tif
            NamePattern = dict()
            filename = re.split("c1xy1t|"+args.format, filename)
            NamePattern[0] =  filename[0]+'c1xy1t'
            NamePattern[1] = args.format+filename[-1]
            NamePattern[2] = len(filename[1])
        else:
            print("Unexpected filename pattern")
            print(filename)
            exit(0)

    if NamePattern[2] ==4:
        Image_Sequence = folder_path+"\\"+NamePattern[0]+'%04d'+NamePattern[1]
    elif NamePattern[2] ==3 :
        Image_Sequence = folder_path+"\\"+NamePattern[0]+'%03d'+NamePattern[1]
    capture = cv.VideoCapture(Image_Sequence)
elif args.input:
    capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
else:
    print("Input not given")
    exit(0)


if not capture.isOpened():
    if args.image_folder:
        print('Unable to open: ' + args.image_folder)
    elif args.input:
        print('Unable to open: ' + args.input)
    exit(0)


while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    fgMask = backSub.apply(frame)
    ret,fgMask = cv.threshold(fgMask, 5, 255, cv.THRESH_BINARY)
    
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    # cv.imshow('Frame', frame)
    # cv.imshow('FG Mask', fgMask)
    if args.image_folder:
        cv.imwrite(os.path.join(bgsub_folder, image_name_list[int(capture.get(cv.CAP_PROP_POS_FRAMES))-1].replace('.tif', '_bg.tif')), fgMask, [cv.IMWRITE_TIFF_COMPRESSION,1], ) 
    
    # keyboard = cv.waitKey(30)
    # if keyboard == 'q' or keyboard == 27:
    #     break

#video generation
os.system(f"python.exe C:\\Users\\cotton\\OneDrive\\XF_Lab\\Trajectory\\TIF2AVI\\tif2avi.py -t .tif -f {bgsub_folder}")