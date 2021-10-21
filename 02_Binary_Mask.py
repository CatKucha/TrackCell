#Cotton Z 2021.02.08
import argparse
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage.io
import skimage.measure


parser = argparse.ArgumentParser(description="Crop image to channel only")
parser.add_argument('--image_folder', '-f', required=True, dest='image_folder', help='the absolute path of the image directory')
parser.add_argument('--threshold', '-t', dest='threshold', type=int, required=False)
parser.add_argument('--channel_area', '-area', dest='channel_area', type=int, required=False, help="The minimum area that a channel should be")
parser.add_argument('--reduced_height', '-rh', dest='reduced_height', type=int, default=0)
parser.add_argument('--reduced_width', '-rw', dest='reduced_width', type=int, default=2) #!!!
parser.add_argument('--minwidth', '-minw', dest='minwidth', type=int, default=14) #must be even number
parser.add_argument('--maxwidth', '-maxw', dest='maxwidth', type=int, default=24)
args = parser.parse_args()

if __name__ == "__main__":
    ######################################################
    #Folder initialization -- Make Other_Files folder
    ######################################################
    args.image_folder = args.image_folder.rstrip('/').rstrip('\\')
    print ("The original image folder is "+args.image_folder)
    OtherFolder = args.image_folder+'/Other_Files'
    if not os.path.isdir(OtherFolder):
        os.mkdir(OtherFolder)


    ######################################################
    #Get the very first image mask
    ######################################################
    #Read folder
    image_fn_list = []
    for file in os.listdir(args.image_folder):
        if '_cropped.tif' in file:
            image_fn_list.append(file)
    if len(image_fn_list)== 0:
        print("No TIFF image in this folder! exist!")
        sys.exit()
    #Read the first image
    first_image = skimage.io.imread(os.path.join(args.image_folder, image_fn_list[0]))

    #first image grayscale histogram
    if args.threshold:
        pass
    else:
        grayscale_flatten = np.ndarray.flatten(first_image)
        plt.hist(grayscale_flatten, bins=range(256))
        plt.title("Raw image grayscale histogram")
        plt.show()
        args.threshold = input("Enter the thresholding value (must be an integer range from 0-255): ")
        while True:
            try:
                args.threshold = int(args.threshold)
                if 0<=args.threshold <=255:
                    break
                else:
                    args.threshold = input("Enter the thresholding value (must be an integer range from 0-255): ") 
            except:
                print('Input is not an integer, please try again')
                args.threshold = input("Enter the thresholding value (must be an integer range from 0-255): ")


    threshold_mask = cv2.inRange(first_image, 0,args.threshold ) #threshold 90 ??
    plt.imshow(threshold_mask, cmap='gray')
    plt.title("After Thresholding")
    plt.show()
    # Perform CCA on the mask
    labeled_image,a = skimage.measure.label(threshold_mask, connectivity=2, return_num=True)
    # convert the label image to color image
    colored_label_image = skimage.color.label2rgb(labeled_image, bg_label=0)
    plt.imshow(colored_label_image)
    plt.title("After Component Conectivity Analysis")
    plt.show()


    #Labeled objects feature
    object_features  = skimage.measure.regionprops(label_image=labeled_image )

    #Object area histogram
    if args.channel_area:
        pass
    else:
        object_areas = [i["area"] for i in object_features]
        plt.hist(object_areas, bins=range(0, 10000, 100))
        plt.title("Object Areas Histogram")
        plt.show()
        args.channel_area = input("Enter the area cutoff value (must be an integer): ")
        while True:
            try:
                args.channel_area = int(args.channel_area)
                break
            except:
                print('Input is not an integer, please try again')
                args.channel_area = input("Enter the area cutoff value (must be an integer): ")

 
    #Generate final mask
    Channel_count = 0
    reduced_rec_mask = np.full(labeled_image.shape, False)
    for object in object_features: 
        min_row, min_col, max_row, max_col = object["bbox"]
        if (args.channel_area<object["area"] <5000) and ((max_col-min_col) <50):
            if max_col-min_col <=args.minwidth: #user defined cutoff! ??!!  let the narrow channel become larger, let the wide channel become smaller
                # print('catch')
                mean = round((max_col+min_col)/2)
                max_col = int(mean+args.minwidth/2)
                min_col = int(mean-args.minwidth/2)
            elif max_col-min_col >= args.maxwidth:
                mean = round((max_col+min_col)/2)
                # print('catch')
                max_col = int(mean+args.maxwidth/2)
                min_col = int(mean-args.maxwidth/2)
            reduced_rec_mask[args.reduced_height:labeled_image.shape[0]-args.reduced_height, min_col+args.reduced_width:max_col- args.reduced_width ] = True
            Channel_count += 1
    print("The total number of detected channel is " +str(Channel_count))
    plt.imshow(reduced_rec_mask, cmap='gray')
    plt.title("Final Channel Mask")
    plt.show()


    #Test the first image after masking
    plt.subplot(211)
    first_image_temp = first_image.copy()
    first_image_temp[reduced_rec_mask==False] =0
    plt.imshow(first_image_temp, cmap='gray')
    plt.title("TEST! Masking the first image")
    plt.subplot(212)
    plt.imshow(first_image, cmap='gray')
    plt.title("the first raw image")
    plt.show()


    #Save Binary Mask File
    reduced_rec_mask=skimage.img_as_ubyte(reduced_rec_mask)
    skimage.io.imsave(fname=OtherFolder+"/Binary_Mask.tif" ,arr=reduced_rec_mask)

    #Write down log file
    with open(OtherFolder +'/mask.log', 'w') as outputfile:
        outputfile.write(f"Source image folder: {args.image_folder}\n")
        outputfile.write(f"mask image thresholding value: {args.threshold}\n")
        outputfile.write(f"mask image channel area cutoff value: {args.channel_area}\n")
        outputfile.write(f"mask image channel reduced_height value: {args.reduced_height}\n")
        outputfile.write(f"mask image channel reduced_width value: {args.reduced_width}\n")


