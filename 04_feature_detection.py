#Cotton Z 2021.03.18
'''
Home made feature detection script ;)
'''

import argparse
import collections
import os
import sys
import cv2
import matplotlib.pyplot as plt
import skimage.io
import skimage.measure
import re
from collections import defaultdict
import progressbar
import random

parser = argparse.ArgumentParser(description='Home made feature detection, please enjoy it')
parser.add_argument('--image_folder', '-f', required=False, dest='image_folder', help='the path to the channel cropped image directory, not the backgroun subtraction folder')
parser.add_argument('--image', '-i', required=False, dest='image_path', help='the path to the image, used in test mode')
parser.add_argument('--cell_area', '-area', dest='cell_area', default=250, type=int, required=False, help="The minimum area that a cell should be")
parser.add_argument('--output_file', '-o', dest='detection_output', required=False, help="Save the detection result")

args = parser.parse_args()

######################################################
#Some Functions
######################################################
def thin_row_filter(image_to_be_filterd, object_features):
    '''
    filter narrow rows in object gained from object_features. Because they might be the vestige of noise (light channel)
    '''
    for object in object_features:
        row_dict = defaultdict(list) #store the pixel coords in each row
        for (row,col) in object["coords"]:
            row_dict[row].append([row,col])
        for row, coord_list in row_dict.items():
            if len(coord_list)<=3: #at least a cell should be not too thin  #??!!!
                for (row,col) in coord_list:
                    image_to_be_filterd[row,col] = 0
    return image_to_be_filterd


def object_area_shape_filter(labeled_image, object_features, area_cutoff):
    '''
    This function can filter small size object or strange shape object in object_features obtained by labeled_image 
    and turn these objects' label to background in the labeled_image
    '''
    #Cell Area filtering
    for object in object_features:
        if object["area"] < area_cutoff:
            for (row,col) in object["coords"]:
                labeled_image[row, col] = 0
    #Cell shape filtering
    for object in object_features:
        (min_row, min_col, max_row, max_col) = object["bbox"]
        if (max_row-min_row>=150) or (max_row-min_row<=4) or (max_col-min_col<=4) or (max_row-min_row)/(max_col-min_col) >= 4: #???!!!
            for (row,col) in object["coords"]:
                labeled_image[row, col] = 0
    #cell shape filtering - second round - filter sticks
    for object in object_features:
        row_dict = defaultdict(int) #store the pixel number in each row
        col_dict = defaultdict(int)  #store the pixel number in each column
        for (row,col) in object["coords"]:
            row_dict[row] += 1
            col_dict[col] += 1
        total_row = len(row_dict)
        total_col = len(col_dict)
        row_less_than_5_count = 0
        col_less_than_5_count = 0
        for row, count in row_dict.items():
            if count <= 4: #!!!???
                row_less_than_5_count += 1
        for col, count in col_dict.items():
            if count <= 4: #!!!???
                col_less_than_5_count += 1
        if col_less_than_5_count/total_col >= 0.6 or row_less_than_5_count/total_row>=0.6:
            for (row,col) in object["coords"]:
                labeled_image[row, col] = 0
                print("filtered one by the ratio!") #??!!
    return labeled_image


def filter_allinone_after_firstCCA(image_final, object_features_obtained_from_image_final):
    image_filterd = image_final.copy()
    #Thin line filtering
    image_filterd = thin_row_filter(image_filterd, object_features_obtained_from_image_final)
    #Dilation
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5), (2,2))
    image_filterd = cv2.dilate(image_filterd, element)
    #Perform CCA 
    labeled_image_filterd, temp = skimage.measure.label(image_filterd, connectivity=2, return_num=True)
    object_features_filtered = skimage.measure.regionprops(label_image=labeled_image_filterd )
    #Cell area and shape filtering
    labeled_image_filterd = object_area_shape_filter(labeled_image_filterd, object_features_filtered, args.cell_area) #attention! object_features_filtered is generated before the final labeled_image_filterd
    return(labeled_image_filterd, object_features_filtered)




######################################################
#Folder Initialization
######################################################
#modify the image folder path
if args.image_folder:
    args.image_folder = args.image_folder.rstrip('/').rstrip('\\')
    bgsub_folder = args.image_folder +'\\bg_subtraction'
    print ("the background subtraction image folder is: "+bgsub_folder+'\n', flush=True)
    #check Other_Files directory existence
    OtherFolder = args.image_folder+'\\Other_Files' #Need to save the result to it
    if not os.path.isdir(OtherFolder):
        print("The OtherFolder does not exist! Please run 02_Binary_Mask.py first! Exiting...")
        exit(0)
    if not os.path.isdir(bgsub_folder):
        print("The backgroud folder does not exist! Please run 03.py first! Exiting...")
        exit(0)


#########################################################
#Read images
#########################################################
image_fn_list = []
if args.image_folder:
    for file in os.listdir(bgsub_folder):
        if '_bg.tif' in file:
            image_fn_list.append(file)
    if len(image_fn_list)== 0:
        print("No TIFF image in this folder! exist!")
        exit()
    mask = skimage.io.imread(OtherFolder+"\\Binary_Mask.tif")
elif args.image_path:
    print(args.image_path.split('\\'))
    bgsub_folder = ('\\').join(args.image_path.split('\\')[:-1]) #in order to match the variable name used in image_folder part
    image_fn_list.append(args.image_path.split('\\')[-1])
    mask = skimage.io.imread(bgsub_folder+"\\Binary_Mask.tif")

#########################################################
#Test the first image for the area cutoff
#########################################################
random_pick = random.randint(1, len(image_fn_list)) -1 
print(f"the random picked image is number {random_pick} image, please check the original image if needed")
first_image_fn = image_fn_list[random_pick]
#Read Image
image = skimage.io.imread(os.path.join(bgsub_folder,first_image_fn))
#Binary Mask filtering
image_masked = image.copy()
image_masked[mask==0] = 0
#Dilation
element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (1,1))
image_dilation = cv2.dilate(image_masked, element)
#Erosion
element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (1,1))
image_dilation_erosion = cv2.erode(image_dilation, element)
image_final = image_dilation_erosion
#Perform CCA 
labeled_image, temp = skimage.measure.label(image_final, connectivity=2, return_num=True)
object_features = skimage.measure.regionprops(label_image=labeled_image )
colored_label_image = skimage.color.label2rgb(labeled_image, bg_label=0)

# #check the cell area suitable or not
# while True:
#     plt.figure()
#     #First Figure - Plot raw figure
#     plt.subplot(411)
#     plt.imshow(image, cmap="gray")
#     plt.title(("Raw binary image"))

#     #Second Figure - Plot colored labeled image 
#     plt.subplot(412)
#     plt.imshow(colored_label_image)
#     plt.title(("After Component Conectivity Analysis (Dialate + Erosion)"))

#     ##########Filtering
#     labeled_image_filterd, object_features_filtered = filter_allinone_after_firstCCA(image_final, object_features_obtained_from_image_final=object_features)

#     #Third Figure - Plot histogram of object size distribution
#     plt.subplot(413)
#     object_areas = [i["area"] for i in object_features_filtered]
#     plt.hist(object_areas, bins=range(0,1000,20))
#     plt.title("Object Areas Histogram")

#     #Third Figure - Plot the final result
#     plt.subplot(414)
#     colored_label_image_filtered = skimage.color.label2rgb(labeled_image_filterd, bg_label=0)
#     plt.imshow(colored_label_image_filtered)
#     plt.title(f"Filtered cell area that below {args.cell_area}")
#     plt.show()

#     val = input("Cell Minimum Area Suitable? [Y]/[N]")
#     if val.lower() == "y" or val.lower() == "yes":
#         break
#     elif val.lower() == "n" or val.lower() == "no":
#         args.cell_area = input("Enter the cell area cutoff value (must be an integer): ")
#         args.cell_area = int(args.cell_area)
#     else:
#         print("Input should be yes or no")
#         print("Exiting..........")
#         sys.exit()



###########################################
#Output Save
###########################################
spot_ID = 0
if args.detection_output:
    output_file = open(args.detection_output, 'w')
else:
    if os.path.isfile(OtherFolder+'/Spot_Detection_Result.tsv'):
        val = input("Spot_Detection_Result.tsv exist, Overwrite? [Y]/[N]")
        if val.lower() == "n" or val.lower() == "no":
            print("Exiting..........")
            exit(0)
        elif val.lower() == "y" or val.lower() == "yes":
            output_file = open(OtherFolder+'/Spot_Detection_Result.tsv', 'w')
            output_file.write('\t'.join(["SPOT_ID", "FRAME", "POSITION_X", "POSITION_Y" ])+'\n')
        else:
            print("Input should be yes or no")
            print("Exiting..........")
            exit(0)
    else:
        output_file = open(OtherFolder+'/Spot_Detection_Result.tsv', 'w')
        output_file.write('\t'.join(["SPOT_ID", "FRAME", "POSITION_X", "POSITION_Y" ])+'\n')




######################################################
#Read images and process them
######################################################
PGbar = progressbar.ProgressBar()
PGbar.start(len(image_fn_list))
count = 0
for image_fn in image_fn_list:
    #Read Image
    image = skimage.io.imread(os.path.join(bgsub_folder,image_fn))
    #Binary Mask filtering
    image_masked = image.copy()
    image_masked[mask==0] = 0
    #Dilation
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (1,1))
    image_dilation = cv2.dilate(image_masked, element)
    #Erosion
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (1,1))
    image_dilation_erosion = cv2.erode(image_dilation, element)
    image_final = image_dilation_erosion
    #Perform CCA 
    labeled_image, temp = skimage.measure.label(image_final, connectivity=2, return_num=True)
    object_features = skimage.measure.regionprops(label_image=labeled_image)
    #Filtering
    labeled_image_filterd, object_features_filtered = filter_allinone_after_firstCCA(image_final, object_features_obtained_from_image_final=object_features)

    #Get the center coordinate of each cell/spot
    object_features = skimage.measure.regionprops(label_image=labeled_image_filterd )
    cell_center_x_list = []
    cell_center_y_list = []
    for object in object_features:
        y,x = object["centroid"]
        cell_center_x_list.append(round(x))
        cell_center_y_list.append(round(y))

    # #Test Program
    # #Plot colored labeled image 
    # plt.figure()
    # plt.subplot(311)
    # plt.imshow(colored_label_image)
    # plt.title(("After Component Conectivity Analysis"))
    # #Plot histogram of object size distribution
    # plt.subplot(312)
    # object_areas = [i["area"] for i in object_features]
    # plt.hist(object_areas, bins=range(0,1500,20))
    # plt.title("Object Areas Histogram")
    # #Plot fintered colored labeled image 
    # plt.subplot(313)
    # labeled_image_filterd = labeled_image.copy()
    # for object in object_features:
    #     if object["area"] < args.cell_area:
    #         for (row,col) in object["coords"]:
    #             labeled_image_filterd[row, col] = 0
    # colored_label_image_filtered = skimage.color.label2rgb(labeled_image_filterd, bg_label=0)
    # plt.imshow(colored_label_image_filtered)
    # plt.title(f"Filtered cell area that below {args.cell_area}")
    # plt.show()


    # #Draw the center of the cell in the figure
    # fig, ax = plt.subplots()
    # # plt.axis('off')
    # plt.imshow(labeled_image_filterd)
    # plt.scatter(cell_center_x_list, cell_center_y_list, c='r', s=1)
    # plt.show()

    #write down the detection result

    filename = image_fn
    if 'c1t'in filename and 'xy1' in filename and "xy1c1" not in filename:
        filename = re.split("c1t|xy1", filename)
    elif "c1t" in filename and 'xy1' not in filename:
        filename = re.split("c1t|_cropped", filename) 
    elif "xy1c1" in filename: #0--10x--flic(mg1655)--200-1034t0001xy1c1_cropped
        experiment_index = args.image_folder.split('/')[-1].split('\\')[-1].replace("-_Channel_only", '').rstrip('-').split('-')[-1]
        repattern = experiment_index+'t|xy1c1'
        filename = re.split(repattern, filename)
    elif "c1" in filename and "c1t" not in filename and "xy1" not in filename: #0--10x-mg1655--200-1048t0001c1_cropped.tif
        experiment_index = args.image_folder.split('/')[-1].split('\\')[-1].replace("_Channel_only", '').rstrip('-').split('-')[-1]
        repattern = experiment_index+'t|c1_cropped'
        filename = re.split(repattern, filename)
    else:
        print("Unexpected filename pattern")
        print(filename)
        exit(0)
    frame_number = int(filename[1]) -1 







    for i in range(len(cell_center_x_list)):
        output_file.write('\t'.join([str(spot_ID), str(frame_number), str(cell_center_x_list[i]), str(cell_center_y_list[i]) ])+'\n')
        spot_ID += 1
    
    count+=1
    PGbar.update(count)

PGbar.finish()
output_file.close()


#Save used parameters
with open (OtherFolder+'/feature_detection_parameter.log', 'w') as output_file:
    output_file.write(str(args.cell_area)+'\n')




#     image_filterd = image_final.copy()
#     #Thin line filtering
#     image_filterd = thin_row_filter(image_filterd, object_features)

#     #Dilation
#     element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5), (2,2))
#     image_filterd = cv2.dilate(image_filterd, element)
#     #Perform CCA 
#     labeled_image_filterd, temp = skimage.measure.label(image_filterd, connectivity=2, return_num=True)
#     object_features_filtered = skimage.measure.regionprops(label_image=labeled_image_filterd )
#     #Cell area and shape filtering
#     labeled_image_filterd = object_area_shape_filter(labeled_image_filterd, object_features_filtered, args.cell_area)
