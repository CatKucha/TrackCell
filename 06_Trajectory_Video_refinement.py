#Cotton Z 2021.03.01
'''
After linking step, we need to test the effectiveness of our parameters, like max_frame_gap, max_move_same_trend ... So this script is used for producing cell trajectory video to help us to modify those parameters
'''

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image
import os
import numpy as np
from matplotlib.collections import LineCollection
from shutil import rmtree
import progressbar
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib as mpl


parser = argparse.ArgumentParser(description="add a trajectory layer to original image")
parser.add_argument('--tracks', '-t', required=False, dest='tracks_file', help='spot-tracks linking file')
parser.add_argument('--image_folder', '-i', required=True, dest='image_folder', help='the path of the image directory')
parser.add_argument('--mode','-mode', dest='mode', default='full_trajectory',choices=['full_trajectory', 'double20frames', 'full', 'part'])

args = parser.parse_args()

###########################################
#Initializtion, Read test image
###########################################
#modify the image folder path
if args.image_folder[-1] == '/':
    args.image_folder = args.image_folder.rstrip('/')
elif args.image_folder[-1] == '\\':
    args.image_folder = args.image_folder.rstrip('\\')
args.image_folder = args.image_folder.replace('"','')
print ("the image folder is: "+args.image_folder, flush=True)

#generate trajectory folder
trajectory_layer_image_folder = args.image_folder + "/trajectory_layer"
if os.path.isdir(trajectory_layer_image_folder):
    val = input("Directory exist, Overwrite? [Y]/[N]")
    if val.lower() == "y" or val.lower() == "yes":
        rmtree(trajectory_layer_image_folder)
        os.mkdir(trajectory_layer_image_folder)
    elif val.lower() == "n" or val.lower() == "no":
        print("Exiting..........")
        exit(0)
    else:
        print("Input should be yes or no")
        print("Exiting..........")
        exit(0)
else:
    os.mkdir(trajectory_layer_image_folder)

#read the test image
findone = False
for first_image in  os.listdir(args.image_folder):
  if '.tif' in first_image and '_sbg.tif' not in first_image:
    findone = True
    break
if findone== False:
  print("No TIFF image in this folder! exist!")
  exit(0)

first_image = matplotlib.image.imread(os.path.join(args.image_folder, first_image))
imgHeight, imgWidth = first_image.shape


###########################################
#Read Tracks
###########################################
if args.tracks_file:
  tracks_pd = pd.read_csv(args.tracks_file, sep='\t', header=0)
else:
  args.tracks_file =args.image_folder+'/Other_Files/Linking_Result.tsv'
  tracks_pd = pd.read_csv(args.tracks_file, sep='\t', header=0)
tracks_pd["POSITION_Y"] = imgHeight - tracks_pd["POSITION_Y"]

###########################################
#Some Preparations
###########################################

def draw_a_trajectory_image(image, lines, colors, endpoints_x, endpoints_y  ):
  line_segments = LineCollection(lines, linewidth=0.5,colors=colors, linestyle='solid')
  fig,ax=plt.subplots()
  fig.set_size_inches(16, 12)

  ax.add_collection(line_segments)
  ax.scatter(endpoints_x, endpoints_y, s=10,color=colors)

  for i in range(len(endpoints_x)):
    plt.text(x=endpoints_x[i], y=endpoints_y[i]-10, s=trackID_list[i],family='sans-serif', size=8, color=colors[i])

  ax.imshow(first_image, cmap="gray")

  plt.gca().set_axis_off()
  plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
  plt.margins(0,0)
  plt.savefig("test.jpg", bbox_inches = 'tight', pad_inches = 0, dpi=200)
  plt.show()
  plt.close()


def draw_full_trajectory(tracks_pd, ):
  my_cmap = plt.get_cmap('gist_rainbow')
  lines = []
  colors = []
  endpoints_x = []
  endpoints_y = []
  trackID_list = list(tracks_pd["TRACK_ID"].unique())
  for trackID in tracks_pd["TRACK_ID"].unique():
    df = tracks_pd[tracks_pd["TRACK_ID"]==trackID]
    my_color =  my_cmap(np.random.rand())
    lines.append(  df[['POSITION_X', 'POSITION_Y']]    )
    colors.append(my_color)
    endpoints_x.append(df['POSITION_X'].iloc[-1])
    endpoints_y.append(df['POSITION_Y'].iloc[-1])


#trackID-color-dict
my_cmap = plt.get_cmap('gist_rainbow')
np.random.seed(0)
color_order = np.linspace(0,1,len(tracks_pd["TRACK_ID"].unique()))
np.random.shuffle(color_order)
color_list = plt.get_cmap('gist_rainbow')(color_order)
track_list = list(tracks_pd["TRACK_ID"].unique())
trackid_color_dict = {track_list[i]:color_list[i] for i in range(len(track_list))}



###########################################
#Trajectory Check - test image
###########################################
if args.mode=='full_trajectory'or args.mode=='full':
  my_cmap = plt.get_cmap('gist_rainbow')
  lines = []
  colors = []
  endpoints_x = []
  endpoints_y = []
  trackID_list = list(tracks_pd["TRACK_ID"].unique())
  for trackID in tracks_pd["TRACK_ID"].unique():
    df = tracks_pd[tracks_pd["TRACK_ID"]==trackID]
    my_color =  my_cmap(np.random.rand())
    lines.append(  df[['POSITION_X', 'POSITION_Y']]    )
    colors.append(my_color)
    endpoints_x.append(df['POSITION_X'].iloc[-1])
    endpoints_y.append(df['POSITION_Y'].iloc[-1])

  line_segments = LineCollection(lines, linewidth=0.5,colors=colors, linestyle='solid')

  # endpoints_x=[10,50,500]
  # endpoints_y=[20, 60, 150]
  # colors=colors[0:3]

  fig,ax=plt.subplots()
  fig.set_size_inches(16, 12)

  ax.add_collection(line_segments)
  ax.scatter(endpoints_x, endpoints_y, s=10,color=colors)

  for i in range(len(endpoints_x)):
    plt.text(x=endpoints_x[i], y=endpoints_y[i]-10, s=trackID_list[i],family='sans-serif', size=8, color=colors[i])

  ax.imshow(first_image, cmap="gray")

  plt.gca().set_axis_off()
  plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
  plt.margins(0,0)
  # plt.savefig("test.jpg", bbox_inches = 'tight', pad_inches = 0, dpi=200)
  plt.show()
  plt.close()


  ###########################################
  #Full Trajectory Image Sequence Generation
  ###########################################
  PGbar = progressbar.ProgressBar()
  PGbar.start(len(os.listdir(args.image_folder)))
  progress_count=0
  for image_fn in os.listdir(args.image_folder):
    if '.tif' in image_fn and '_sbg.tif' not in image_fn:
      image = matplotlib.image.imread(os.path.join(args.image_folder, image_fn))
      #add trajectory to this image
      fig,ax=plt.subplots()
      fig.set_size_inches(16, 12)
      # ax.set_xlim(0,imgWidth)
      # ax.set_ylim(0,imgHeight)
      line_segments = LineCollection(lines, linewidth=0.5,colors=colors, linestyle='solid')
      # new_line_segments = copy(line_segments)
      ax.add_collection(line_segments)
      ax.scatter(endpoints_x, endpoints_y, s=10,color=colors)
      for i in range(len(endpoints_x)):
          plt.text(x=endpoints_x[i], y=endpoints_y[i]-10, s=trackID_list[i],family='sans-serif', size=8, color=colors[i])
      ax.imshow(image, cmap="gray")
      plt.gca().set_axis_off()
      plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
      plt.margins(0,0)
      plt.savefig(os.path.join(trajectory_layer_image_folder,image_fn.replace('.tif', '_tracks.jpg') ), bbox_inches = 'tight', pad_inches = 0, dpi=200)
      plt.close()
    progress_count+=1
    PGbar.update(progress_count)

  PGbar.finish()


###########################################
#Double 20 Frames Trajectory Image Sequence Generation
###########################################

elif args.mode=='double20frames'or args.mode=='part':
  PGbar = progressbar.ProgressBar()

  PGbar.start(len(os.listdir(args.image_folder)))
  progress_count = 0

  for image_fn in os.listdir(args.image_folder):
    if '.tif' in image_fn and '_sbg.tif' not in image_fn:
      image = matplotlib.image.imread(os.path.join(args.image_folder, image_fn))

      frame_index = progress_count
      circles = []
      lines = []
      colors = []
      current_frame_xy_list = []
      particel_frame_df = tracks_pd[ tracks_pd["FRAME"] == frame_index ]
      trackID_list = list(particel_frame_df["TRACK_ID"].unique())
      for trackID in trackID_list:
        track_color = trackid_color_dict[trackID]
        circle_xy = particel_frame_df[particel_frame_df["TRACK_ID"]==trackID][ ['POSITION_X','POSITION_Y'] ]
        if len(circle_xy) >1:
          print("ERROR!!")
          exit(0)
        double20_xy_lists = tracks_pd[(tracks_pd["TRACK_ID"]==trackID) & (tracks_pd["FRAME"] >= frame_index-20) & (tracks_pd["FRAME"] <= frame_index+20) ][ ['POSITION_X','POSITION_Y'] ]
        circles.append(Circle((circle_xy['POSITION_X'],circle_xy['POSITION_Y'] ), 10 ,  facecolor='none', fill=False))
        current_frame_xy_list.append(circle_xy)
        lines.append(double20_xy_lists)
        colors.append(track_color)

      #Figure Initialization
      fig,ax=plt.subplots()
      plt.axis('off')
      plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
      plt.margins(0,0)
      fig.set_size_inches(8, 3)
      # ax.set_xlim(0,imgWidth)
      # ax.set_ylim(0,imgHeight)
      
      line_collection = LineCollection(lines, linewidth=0.5,colors=colors, linestyle='solid')
      circle_collection = PatchCollection(circles, edgecolor=colors, lw=0.3, match_original=True)
      ax.add_collection(line_collection)
      ax.add_collection(circle_collection)
      #add text
      for i in range(len(trackID_list)):
        plt.text(x=current_frame_xy_list[i]['POSITION_X']+10, y=current_frame_xy_list[i]['POSITION_Y']+3, s=trackID_list[i],family='sans-serif', size=6, color=colors[i])
      
      ax.imshow(image, cmap="gray")
      
      
      # plt.savefig(os.path.join(trajectory_layer_image_folder,image_fn.replace('.tif', '_tracks.jpg') ), bbox_inches = 'tight', pad_inches = 0, dpi=200)
      plt.savefig(os.path.join(trajectory_layer_image_folder,image_fn.replace('.tif', '_tracks.jpg') ), pad_inches = 0, dpi=300)
      # plt.show()
      plt.close()
      progress_count+=1
      PGbar.update(progress_count)

  PGbar.finish()

      






        

      
      




