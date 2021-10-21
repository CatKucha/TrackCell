#Cotton Z 2020.12.23 
#version 1.0

'''
Crop the microscopic image to the region of channel only
Step 1. automatically rotate the image
Step 2. crop the image
'''

import argparse
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import os 
import sys
from shutil import rmtree
import progressbar



parser = argparse.ArgumentParser(description="Crop image to channel only")
parser.add_argument('--image', '-i', required=False, default=None, dest='original_image', help="microscopy image file (8 bit, gray" )
parser.add_argument('--image_folder', '-f', required=True, dest='image_folder', help='the absolute path of the image directory')
#Canny contour edges of the image
parser.add_argument('--LowerBoundForCanny', '-lower', dest='LowerBound', type=int, default=100, help="this number is the lowest number of grayscale for line detection") 
parser.add_argument('--UpperBoundForCanny', '-upper', dest='UpperBound', type=int, default=200)
#HoughlinesP find lines in the image
parser.add_argument('--minLineLength_for_HoughLinesP', '-minL', dest='minLineLength', type=int, default=100,help='Lines which length below this number are not shown')
parser.add_argument('--maxLineGap_for_HoughLinesP', '-maxG', dest='maxLineGap', type=int, default=7)
#Rotation angle of the image
parser.add_argument('--rotation_direction', '-r', dest="RotationDirection", default="need_clockwise",choices=["need_counterclockwise", "need_clockwise", "need_cntc", "need_c"], help='specify the rotation direction of the original image')
#Do you want to check the result of canny function?
parser.add_argument('--canny_test', '-canny', dest='cannySwitch', help='Show the Canny figure to figure out what happened', action="store_true")
#Do you need the image to be upside down?
parser.add_argument('--upside_down', dest='upsidedown_Switch', help='Flip all the figures because they are upside down', action="store_true")
#the channel height
parser.add_argument('--channel_height', '-Cheight', dest='channel_height', type=int, default=280,help='the height used to unify cropped figure image size across all experiments')
#the center of the channel
parser.add_argument('--channel_center_y', '-centerY', dest='channel_center_y', type=int, help='specify the center y coordinate')

args = parser.parse_args()

def figure_resize(image, ratio):
    height, width = image.shape[:2]
    r_height = int(height*ratio)
    r_width = int(width*ratio)
    dim = (r_width, r_height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def reject_outliers(data, m=1):
    new_data = list()
    data_mean = np.mean(data)
    data_std = np.std(data)
    for i in data:
        if abs(i - data_mean) < (m * data_std):
            new_data.append(i)
    return new_data


def rotation_angle_detection(image):
    ##############
    #Rotate image#
    ##############
    image_copy = image.copy()
    #Detect rotation angle
    img_edges = cv2.Canny(image, args.LowerBound, args.UpperBound, apertureSize=3, L2gradient=False)
    img_lines = cv2.HoughLinesP(img_edges,rho=1, theta=math.pi/180.0, threshold=100, minLineLength=args.minLineLength, maxLineGap=args.maxLineGap) # a threshold of the minimum number of intersections needed to detect a line.
    if args.cannySwitch:
        plt.subplot(121), plt.imshow(image_copy, cmap='gray')
        plt.title("Original image"), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img_edges, cmap='gray')
        plt.title("Edges image"), plt.xticks([]), plt.yticks([])
        plt.show()
    
    #only those horizontal segements are usefull
    angles = []
    for [[x1, y1, x2, y2]] in img_lines:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if args.RotationDirection == "need_clockwise" or args.RotationDirection =="need_c":
            if -90< angle <=0:
                angles.append(angle)
                cv2.line(image_copy, (x1, y1), (x2, y2), (255, 0, 0), 3)
        else: #args.RotationDirection == "need_counterclockwise" or args.RotationDirection =="need_cntc"
            if 0<= angle <90:
                angles.append(angle)
                cv2.line(image_copy, (x1, y1), (x2, y2), (255, 0, 0), 3)
    
    median_angle = np.median(angles)
    
    return (image_copy, median_angle)

def adjust_channel_with_assumption(data):
    '''modify y_axis_list with current knowledge, like the height of the channel, the center of the channel'''
    new_data = list()
    if args.channel_center_y:
        data_mean = args.channel_center_y
    else:
        data_mean = np.mean(data)
    data_mean = int(data_mean+0.5)
    print(f"the y coordinate of the channel center is: {data_mean}")
    new_data.append(int(data_mean+args.channel_height*0.5))
    new_data.append(int(data_mean-args.channel_height*0.5))
    return new_data


def crop_image2channel(image, median_angle):
    if median_angle != 0:
        img_rotated = ndimage.rotate(image, median_angle, reshape=False)
    else:
        img_rotated = image.copy()

    ##############
    #Crop image#
    ##############
    #Find the upper and lower bound of the channel
    img_rotated_copy = img_rotated.copy()
    img_rotated_edges = cv2.Canny(img_rotated, args.LowerBound, args.UpperBound, apertureSize=3, L2gradient=False)
    img_rotated_lines = cv2.HoughLinesP(img_rotated_edges,rho=1, theta=math.pi/180.0, threshold=100, minLineLength=args.minLineLength, maxLineGap=args.maxLineGap) # a threshold of the minimum number of intersections needed to detect a line.

    y_axis_list = []
    for [[x1, y1, x2, y2]] in img_rotated_lines:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if -10< angle <10: #????????~!!!
            y_axis_list.append(y1)
            y_axis_list.append(y2)
            cv2.line(img_rotated_copy, (x1, y1), (x2, y2), (255, 0, 0), 3)
    if median_angle != 0:
        y_axis_list = reject_outliers(y_axis_list, m=1)
    else:
        y_axis_list = reject_outliers(y_axis_list, m=1.5)
    y_axis_list = adjust_channel_with_assumption(y_axis_list)
    y_max = max(y_axis_list)
    y_min = min(y_axis_list)
    x_max = img_rotated.shape[1]
    crop_img = img_rotated[y_min:y_max, 0:x_max].copy()

    return  img_rotated_copy, crop_img, y_max, y_min, x_max







if __name__ == "__main__":
    #check folder path
    if args.image_folder[-1] == '/':
        args.image_folder = args.image_folder.rstrip('/')
    elif args.image_folder[-1] == '\\':
        args.image_folder = args.image_folder.rstrip('\\')
    args.image_folder = args.image_folder.replace('"','')

    #check argument
    if args.channel_height%2 == 1:
        print("Please enter a even number as the channel height! not odd number")
    #create cropped image folder first
    # args.image_folder
    #Trim the folder
    aimed_folder_name = os.path.split(args.image_folder)[-1].replace(' ','').replace('"','').rstrip('-')+"_Channel_only"
    #generate the cropped channel image folder in anoher folder
    cropped_folder = os.path.join("D:/Trajectory_Project_Channel", aimed_folder_name)
    # cropped_folder = args.image_folder.replace('"','').replace('(','').replace(')','') +"_Channel_only" #generate the folder in the same parent directory


    if os.path.isdir(cropped_folder):
        val = input("Directory exist, Overwrite? [Y]/[N]")
        if val.lower() == "y" or val.lower() == "yes":
            filelist = [f for f in os.listdir(cropped_folder)]
            for f in filelist:
                try:
                    os.remove(os.path.join(cropped_folder, f))
                except:
                    rmtree(os.path.join(cropped_folder, f))
        elif val.lower() == "n" or val.lower() == "no":
            print("Exiting..........")
            sys.exit()
        else:
            print("Input should be yes or no")
            print("Exiting..........")
            sys.exit()
    else:
        os.mkdir(cropped_folder)


    ##############
    #test figure
    ##############
    if args.original_image == None:
        findone = False
        for first_image in  os.listdir(args.image_folder):
            if '.tif' in first_image and '_sbg.tif' not in first_image:
                findone = True
                break
        if findone== False:
            print("No TIFF image in this folder! exist!")
            sys.exit()
        image = cv2.imread(os.path.join(args.image_folder, first_image))
        # image = skimage.io.imread(os.path.join(args.image_folder, first_image))
        # print(image[0])
    else:
        image = cv2.imread(args.original_image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #if turn on, the image could not have red lines
    if args.upsidedown_Switch:
        # image = cv2.flip(image, 0) #this is not the actual upsidedown in reality
        image = ndimage.rotate(image, 180, reshape=False)
    image_copy, median_angle=rotation_angle_detection(image)
    img_rotated, crop_img, y_max, y_min, x_max = crop_image2channel(image, median_angle)


    print(f"Angle is {abs(median_angle):.04f}. {args.RotationDirection}")
    plt.subplot(131), plt.imshow(image_copy, cmap='gray')
    plt.title("Original image (Detected lines)"), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(img_rotated, cmap='gray')
    plt.title('Rotated image'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(crop_img, cmap='gray')
    plt.title('Cropped image'), plt.xticks([]), plt.yticks([])
    plt.show()


    ##############
    #Crop Image Sequence#
    ##############
    PGbar = progressbar.ProgressBar()
    PGbar.start(len(os.listdir(args.image_folder)))
    count = 0
    for raw_image in os.listdir(args.image_folder):
        if ".tif" in raw_image and '_sbg.tif' not in raw_image:
            image = cv2.imread(os.path.join(args.image_folder, raw_image))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if args.upsidedown_Switch:
                # image = cv2.flip(image, 0)
                image = ndimage.rotate(image, 180, reshape=False)

            img_rotated = ndimage.rotate(image, median_angle, reshape=False)

            crop_img = img_rotated[y_min:y_max, 0:x_max].copy()

            crop_file_name = raw_image.rstrip('.tif')+'_cropped.tif'
            # cv2.imwrite(os.path.join(cropped_folder, crop_file_name), crop_img,  TIFFTAG_COMPRESSION=COMPRESSION_NONE)
            cv2.imwrite(os.path.join(cropped_folder, crop_file_name), crop_img, [cv2.IMWRITE_TIFF_COMPRESSION,1], ) 

        count+=1
        PGbar.update(count)
    
    PGbar.finish()