#Cotton Z 2021.05.28
'''
Use the TrackMate spot detection output table as input, then link all the spots to tracks and plot trajectory figures

what's new

classify tracks into 4 classes "up/down" "fast/slow"

canceled the previous feature: complement the discontinued trajactory

'''
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import maximum
import skimage.io
import skimage.measure
import skimage.viewer
from collections import defaultdict
import logging
import os
import numpy as np
import math
from matplotlib.collections import LineCollection
# import warnings

parser = argparse.ArgumentParser(description="Generate trajectory using spots statistics file")
parser.add_argument('--image_folder', '-f', required=True, dest='image_folder', help='the path to the channel cropped image directory, not the Otherfolder')
parser.add_argument('--spots_file', '-s', dest='spots_file', required=False, help='Spots_in_tracks statistics.tsv')
parser.add_argument('--mask_image', '-m', dest='img_mask', required=False, help="Channel Mask Image")
parser.add_argument('--output_file', '-o', dest='linking_output', required=False, help="Save the linking result")
parser.add_argument('-figure', dest='plot_out',required=False, help='a temporary function', type=str)

args = parser.parse_args()
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# warnings.filterwarnings("ignore")
###########################################
#Basic Parameters
###########################################
max_frame_gap = 20
# max_move_same_trend = 80
# max_move_opposite_trend = 30 #30 pixel
merge_max_distance = 30  #the max distance before the two cells merge into one cell
merge_first_move_max = 30 #the max distance between the cell and its merged form
max_fake_dectection_distance = 15 #the max distance between the correct and wrong detection spots in the same cell

#{currentframe-latestframe:max_distance} In fact 1 means no gap
#if the frame gap >10 and <20, the cell might get stuck in the channel, it is rolling so it cannot be recognize as a spot for a while
same_trend_frame_gap_distance_gap_dict = {1:150, 2:80, 3:80, 4:80, 5:80, 6:80, 7:80, 8:90, 9:80, 10:80, 11:30, 12:30, 13:30, 14:30, 15:30, 16:10, 17:10, 18:10, 19:10, 20:10 } 
opposite_trend_frame_gap_distance_gap_dict = {1:40, 2:40, 3:30, 4:30, 5:30, 6:30, 7:20, 8:20, 9:20, 10:20, 11:20, 12:10, 13:10, 14:10, 15:10, 16:10, 17:10, 18:10, 19:10, 20:10 } 


######################################################
#Folder Parameters
######################################################
args.image_folder = args.image_folder.rstrip('/').rstrip('\\')
print ("the cropped channel image folder is: "+args.image_folder+'\n', flush=True)
#Folder Destination
OtherFolder = args.image_folder+'/Other_Files'
#check Other_Files directory existence
if not os.path.isdir(OtherFolder):
    print("The OtherFolder does not exist! Please run 02_Binary_Mask.py first! Exiting...")
    exit(0)
if not args.spots_file:
    args.spots_file = OtherFolder+'/Spot_Detection_Result.tsv'
if not args.img_mask:
    args.img_mask =  OtherFolder+'/Binary_Mask.tif'
if not args.linking_output:
    args.linking_output = OtherFolder+'/Linking_Result.tsv'


###########################################
#Read binary mask and spot data
###########################################
"""
Output:
xcoord2channel
"""
print("Start to read binary mask and spot data...\n......\n", flush=True)
mask_img = skimage.io.imread(args.img_mask)
imgHeight, imgWidth = mask_img.shape
labeled_mask,a = skimage.measure.label(mask_img, connectivity=2, return_num=True)
channel_features = skimage.measure.regionprops(label_image=labeled_mask )

xcoord2channel = dict()
channel_total_index=0

for channel in channel_features:
    min_row, min_col, max_row, max_col = channel['bbox']
    max_col = int(max_col)
    min_col = int(min_col)
    for x in range(min_col-1, max_col+2, 1): #????!!! 
        xcoord2channel[x] = channel_total_index
    channel_total_index += 1
print("The total number of channels is "+str(len(channel_features)), flush=True)
print("Binary mask and spot data reading finished\n", flush=True)

###########################################
#Read TrackMate Spots Statistics Data
###########################################
"""
Output:
channel_frame_spot_dict
"""
print("Read Spots Detection Data...\n", flush=True)
spots_data = pd.read_csv(args.spots_file, sep='\t', header=0 )
spots_data['POSITION_Y'] = imgHeight - spots_data['POSITION_Y']
total_frames = spots_data['FRAME'].max()
total_frames = 1800


#Channel frame spot dict
channel_frame_spot_dict = dict()
for channel_index in range(channel_total_index):
    channel_frame_spot_dict[channel_index] = dict()

outside_count = 0

for index, row in spots_data.iterrows():
    frame = int(row['FRAME'])
    x_position = int(row['POSITION_X'])
    y_position = int(row['POSITION_Y'])
    spot_ID = row['SPOT_ID']
    try:
        channel_index = xcoord2channel[x_position]
        if frame in channel_frame_spot_dict[channel_index]:
            channel_frame_spot_dict[channel_index][frame].append([spot_ID, x_position, y_position])
        else:
            channel_frame_spot_dict[channel_index][frame] = list()
            channel_frame_spot_dict[channel_index][frame].append([spot_ID, x_position, y_position])
    except:
        # print("Warning! x postition outside the channel")
        outside_count+= 1

print("the outside channel spot count is: " +str(outside_count), flush=True)
print("Spot Detection Data Reading Finished\n", flush=True)


#####################################################
#Linking Spots - preprocessing - filter channel >=3 spots occured at the same time
#####################################################
#channel only have <=2 cell coocuccurred
channel_spots_max_number_dict = dict()
channel_frame_cellnumber_count_dict = dict()
for channel in channel_frame_spot_dict:
    channel_frame_cellnumber_count_dict[channel] = {1:0, 2:0,3:0,4:0, 5:0} #if one frame one channnel countain 3 cells then dict[3] add one count
# for channel in channel_frame_spot_dict:
if len(spots_data) != 0:
    #fill out the channel_frame_spot_dict
    for channel in channel_frame_spot_dict:
        for frame in channel_frame_spot_dict[channel]:
            cell_number = len(channel_frame_spot_dict[channel][frame])
            if cell_number >=5:
                cell_number = 5
            channel_frame_cellnumber_count_dict[channel][cell_number] += 1
    #fill out the channel_spots_max_number_dict
    for channel in channel_frame_spot_dict:
        if channel_frame_cellnumber_count_dict[channel][4] >0 or channel_frame_cellnumber_count_dict[channel][5]>0:
            channel_spots_max_number_dict[channel] = 4
        elif channel_frame_cellnumber_count_dict[channel][3] > total_frames*0.01: #???!!!
            channel_spots_max_number_dict[channel] = 3
        elif channel_frame_cellnumber_count_dict[channel][2] >0 :
            channel_spots_max_number_dict[channel] = 2
        elif channel_frame_cellnumber_count_dict[channel][1] >0 :
            channel_spots_max_number_dict[channel] = 1
        elif channel_frame_cellnumber_count_dict[channel][1] ==0 :
            channel_spots_max_number_dict[channel] = 0
        else:
            print("Something Wrong!!")
            print(channel_frame_cellnumber_count_dict)
            print(channel_frame_cellnumber_count_dict[channel])
            exit()

        # for frame in range(total_frames+1): 
        #     if frame in channel_frame_spot_dict[channel]:
        #         if channel not in channel_spots_max_number_dict:
        #             channel_spots_max_number_dict[channel] = len(channel_frame_spot_dict[channel][frame])
        #         elif len(channel_frame_spot_dict[channel][frame]) > channel_spots_max_number_dict[channel]:
        #             channel_spots_max_number_dict[channel] = len(channel_frame_spot_dict[channel][frame])

channel_number_atleast_has_spot = len(channel_spots_max_number_dict)
channel_number_nomore_than_aimed = 0
channel_filtered_list = list()
channel_max_one = 0
channel_max_two = 0
for channel, maxcell in channel_spots_max_number_dict.items():
    if maxcell==2 or maxcell == 1: #!!!???
        channel_number_nomore_than_aimed += 1
        channel_filtered_list.append(channel)
for channel, maxcell in channel_spots_max_number_dict.items():
    if maxcell == 1:
        channel_max_one+=1
    if maxcell == 2:
        channel_max_two += 1

print(f"total channel number is: {len(channel_features)}, \n \
        channel with signals number is: {channel_number_atleast_has_spot}, \n \
        channel has no more than 1 spots in one channel is: {channel_number_nomore_than_aimed}", flush=True)



#####################################################
#Linking Spots - Round One - link spots to tracks
#####################################################
print("Linking Spots - Round One - link spots to tracks...", flush=True)


def update_latest_frame(specific_track_index): 
    '''
    update channel_latestframe_track dict, modify it with specific_track_index
    '''
    global channel_latestframe_track
    global channel
    global frame
    if channel in channel_latestframe_track:
        if channel_latestframe_track[channel][0] == frame:
            channel_latestframe_track[channel].append(specific_track_index)
        elif channel_latestframe_track[channel][0] < frame:
            channel_latestframe_track[channel]=[frame, specific_track_index]
    else:
        channel_latestframe_track[channel]=[frame, specific_track_index]

def update_exist_track(specific_track_index, spot_ID, x_position, y_position):
    '''
    add the current spot to an exist track
    '''
    global channel_track_spot_dict
    global channel
    global frame
    channel_track_spot_dict[channel][specific_track_index].append([frame,spot_ID, x_position, y_position])
    update_latest_frame(specific_track_index)


def add_a_new_track(spot_ID, x_position, y_position):
    '''
    add a new track to the current frame current channel
    '''
    global channel_track_spot_dict
    global channel_latestframe_track
    global channel
    global frame
    global track_index
    channel_track_spot_dict[channel][track_index]=list()
    channel_track_spot_dict[channel][track_index].append([frame,spot_ID, x_position, y_position])
    update_latest_frame(track_index)
    track_index += 1


def add_two_new_tracks(spot_ID_1, x_position_1, y_position_1, spot_ID_2, x_position_2, y_position_2):
    '''
    track_dict -> channel_track_spot_dict  (defaultdict(dict))
    latestframe_dict -> channel_latestframe_track
    add two new tracks to the current frame current channel
    '''
    global channel_track_spot_dict
    global channel_latestframe_track
    global channel
    global frame
    global track_index
    #the first one
    channel_track_spot_dict[channel][track_index]=list()
    channel_track_spot_dict[channel][track_index].append([frame,spot_ID_1, x_position_1, y_position_1])
    update_latest_frame(track_index)
    track_index = track_index +1
    #the second one
    channel_track_spot_dict[channel][track_index]=list()
    channel_track_spot_dict[channel][track_index].append([frame,spot_ID_2, x_position_2, y_position_2])
    update_latest_frame(track_index)
    track_index = track_index +1

def spot_track_distributor(wanted_spot_ID, wanted_track_ID, spot_ID_xy_dict):
    tracks_list = list(spot_ID_xy_dict.keys())
    tracks_list.remove(wanted_spot_ID)
    new_track_spot_ID = tracks_list[0]
    wanted_spot_x_position, wanted_spot_y_position = spot_ID_xy_dict[wanted_spot_ID]
    new_track_spot_x_position, new_track_spot_y_position = spot_ID_xy_dict[new_track_spot_ID]
    update_exist_track(wanted_track_ID, wanted_spot_x_position, wanted_spot_x_position, wanted_spot_y_position)
    add_a_new_track(new_track_spot_ID, new_track_spot_x_position, new_track_spot_y_position)



channel_track_spot_dict = defaultdict(dict)
channel_latestframe_track = dict()
spot_track_dict = dict()
track_index = 0
#Channel by channel
# for channel in channel_frame_spot_dict: 
for channel in channel_filtered_list:
    #frame by frame
    if len(spots_data) != 0:
        for frame in range(total_frames+1): 
            if frame in channel_frame_spot_dict[channel]:
                #1 spot in the frame in the channel
                if len(channel_frame_spot_dict[channel][frame]) == 1:
                    spot_ID, x_position, y_position = channel_frame_spot_dict[channel][frame][0]
                    #the first spot appeared in this channel
                    if channel not in channel_track_spot_dict:
                        add_a_new_track(spot_ID, x_position, y_position)
                    #too large gap between the latestframe and this frame
                    elif frame - channel_latestframe_track[channel][0] > max_frame_gap:
                        add_a_new_track(spot_ID, x_position, y_position)
                    #impossible cells number of the lastest frame
                    elif len(channel_latestframe_track[channel])>3:
                        print("WARNING!!!#Error1 strange number of spots detected at the same channel and the same frame, please recheck the original input data! Existing...")
                        print(f"the channel index is {channel}, the frame is {channel_latestframe_track[channel][0]}")
                        exit(0)
                    #only 1 cell in the lastest frame, 1 spot(could be 2 cells) in the current frame
                    elif len(channel_latestframe_track[channel])==2:
                        latestframe, track1 = channel_latestframe_track[channel]
                        max_move_same_trend = same_trend_frame_gap_distance_gap_dict[frame-latestframe]
                        max_move_opposite_trend = opposite_trend_frame_gap_distance_gap_dict[frame-latestframe]
                        x1, y1 = channel_track_spot_dict[channel][track1][-1][-2:]
                        distance1 = abs(y_position - y1)
                        #if this spot is the second frame of this track
                        if len(channel_track_spot_dict[channel][track1]) == 1:
                            #movement displacement is acceptable
                            if distance1<=max_move_same_trend:
                                update_exist_track(track1, spot_ID, x_position, y_position)
                            else:
                                add_a_new_track(spot_ID, x_position, y_position)
                        #if this spot is the >=3rd frame of this track
                        elif len(channel_track_spot_dict[channel][track1]) >= 2:
                            x2, y2 = channel_track_spot_dict[channel][track1][-2][-2:]
                            if distance1<=max_move_opposite_trend:
                                update_exist_track(track1, spot_ID, x_position, y_position)
                            elif distance1<=max_move_same_trend and (y_position>=y1>=y2 or y_position<=y1<=y2):
                                update_exist_track(track1, spot_ID, x_position, y_position)
                            else:
                                add_a_new_track(spot_ID, x_position, y_position)
                        else:
                            raise ValueError("the length of channel_track_spot_dict[channel][track1] is zero!") #??!!

                    #2 cells in the lastest frame, 1 spot(could be 2 cells) in the current frame
                    #Ending: 2old(merge event) 1old1lost, 1new
                    elif len(channel_latestframe_track[channel])==3:
                        latestframe, track1, track2 = channel_latestframe_track[channel]
                        max_move_same_trend = same_trend_frame_gap_distance_gap_dict[frame-latestframe]
                        max_move_opposite_trend = opposite_trend_frame_gap_distance_gap_dict[frame-latestframe]
                        x1,y1 = channel_track_spot_dict[channel][track1][-1][-2:]
                        x2,y2 = channel_track_spot_dict[channel][track2][-1][-2:]
                        distance1 = abs(y_position - y1)
                        distance2 = abs(y_position - y2)
                        cell_inter_distance = abs(y1 - y2)
                        #the previous two cells have same coordinates
                        if x1==x2 and y1==y2:
                            if distance1<max_move_opposite_trend: #??!!
                                update_exist_track(track1, spot_ID, x_position, y_position)
                                update_exist_track(track2, spot_ID, x_position, y_position)
                            else:
                                add_a_new_track(spot_ID, x_position, y_position)
                        #the previous two cells have different coordinates
                        else:
                            #the current spot is far away from the previous two cells
                            if distance1>max_move_opposite_trend and distance2>max_move_opposite_trend: 
                                add_a_new_track(spot_ID, x_position, y_position)
                            # #fake detection, fake merge event, too short time duration
                            # elif len(channel_track_spot_dict[channel][track1])<=2 or len(channel_track_spot_dict[channel][track2])<=2: #might lost some true signals?
                            #     add_a_new_track(spot_ID, x_position, y_position)
                            #merge event
                            elif cell_inter_distance <= merge_max_distance and distance1<=merge_first_move_max and distance2<=merge_first_move_max and len(channel_track_spot_dict[channel][track1]) >2 and len(channel_track_spot_dict[channel][track2])>2:
                                update_exist_track(track1, spot_ID, x_position, y_position)
                                update_exist_track(track2, spot_ID, x_position, y_position)
                            #1old 1lost
                            elif distance1<distance2:
                                update_exist_track(track1, spot_ID, x_position, y_position)
                            elif distance2<distance1:
                                update_exist_track(track2, spot_ID, x_position, y_position)
                            else:
                                add_a_new_track(spot_ID, x_position, y_position)
                    else:
                        raise Exception('unknown exception happened in 1 spot in the current frame, need further investigation...')




                #2 spots in the current frame
                elif len(channel_frame_spot_dict[channel][frame]) == 2:
                    spot_ID_1, x_position_1, y_position_1 = channel_frame_spot_dict[channel][frame][0]
                    spot_ID_2, x_position_2, y_position_2 = channel_frame_spot_dict[channel][frame][1]
                    #the very first two spots in the channel
                    if channel not in channel_track_spot_dict:
                        add_two_new_tracks(spot_ID_1, x_position_1, y_position_1,spot_ID_2, x_position_2, y_position_2 )
                    #too large gap between the latestframe and this frame
                    elif frame - channel_latestframe_track[channel][0] > max_frame_gap:
                        add_two_new_tracks(spot_ID_1, x_position_1, y_position_1,spot_ID_2, x_position_2, y_position_2 )
                    #impossible cells number of the lastest frame
                    elif len(channel_latestframe_track[channel])>3: 
                        print("WARNING!!!#Error2 strange number of spots detected at the same channel and the same frame, please recheck the original input data! Existing...")
                        print(f"the channel index is {channel}, the frame is {channel_latestframe_track[channel][0]}")
                        exit(0)
                    #1 cell in the latest frame, 2 spots in the current frame, frame gap <= max gap
                    elif len(channel_latestframe_track[channel])==2:
                        latestframe, track1 = channel_latestframe_track[channel]
                        max_move_same_trend = same_trend_frame_gap_distance_gap_dict[frame-latestframe]
                        max_move_opposite_trend = opposite_trend_frame_gap_distance_gap_dict[frame-latestframe]
                        x1,y1 = channel_track_spot_dict[channel][track1][-1][-2:]
                        distance1 = abs(y_position_1 - y1)
                        distance2 = abs(y_position_2 - y1)
                        #if this spot is the second frame of this track, spot2 is a new track
                        if len(channel_track_spot_dict[channel][track1]) == 1:
                            #the new spot is a fake spot, wrong detection
                            if (abs(y_position_1 - y_position_2) <= max_fake_dectection_distance) or ( (abs(y_position_1 - y_position_2) <= max_fake_dectection_distance*2) and ( imgHeight*0.8 <= y_position_1 <= imgHeight*0.8 ) and ( imgHeight*0.8 <= y_position_2 <= imgHeight*0.8 ) ):
                                if distance1<= distance2:
                                    update_exist_track(track1, spot_ID_1, x_position_1, y_position_1)
                                else:
                                    update_exist_track(track1, spot_ID_2, x_position_2, y_position_2)
                            #spot1 belong to the previous track, spot2 is a new track
                            elif distance1<=distance2 and distance1<=max_move_same_trend:
                                update_exist_track(track1, spot_ID_1, x_position_1, y_position_1)
                                add_a_new_track(spot_ID_2, x_position_2, y_position_2)
                            #spot2 belong to the previous track, spot1 is a new track
                            elif distance2<=distance1 and distance2<=max_move_same_trend:
                                update_exist_track(track1, spot_ID_2, x_position_2, y_position_2)
                                add_a_new_track(spot_ID_1, x_position_1, y_position_1)
                            else:
                                add_two_new_tracks(spot_ID_1, x_position_1, y_position_1, spot_ID_2, x_position_2, y_position_2)
                        elif len(channel_track_spot_dict[channel][track1]) >1:
                            x2,y2 = channel_track_spot_dict[channel][track1][-2][-2:]
                            #the new spot is a fake spot, wrong detection
                            if abs(y_position_1 - y_position_2) <= max_fake_dectection_distance or ( (abs(y_position_1 - y_position_2) <= max_fake_dectection_distance*2) and ( imgHeight*0.8 <= y_position_1 <= imgHeight*0.8 ) and ( imgHeight*0.8 <= y_position_2 <= imgHeight*0.8 ) ):
                                if distance1<= distance2:
                                    update_exist_track(track1, spot_ID_1, x_position_1, y_position_1)
                                else:
                                    update_exist_track(track1, spot_ID_2, x_position_2, y_position_2)
                            #spot1 -> track1, spot2 -> new track
                            elif distance1<=distance2 and distance1<=max_move_opposite_trend:
                                update_exist_track(track1, spot_ID_1, x_position_1, y_position_1)
                                add_a_new_track(spot_ID_2, x_position_2, y_position_2)
                            #spot2 -> track1, spot1 -> new track
                            elif distance2<=distance1 and distance2<=max_move_opposite_trend:
                                update_exist_track(track1, spot_ID_2, x_position_2, y_position_2)
                                add_a_new_track(spot_ID_1, x_position_1, y_position_1)
                            #spot1 -> track1, spot2 -> new track
                            elif (distance1<=max_move_same_trend) and (distance1<=distance2) and (y_position_1>y1>y2 or y_position_1<y1<y2 ):
                                update_exist_track(track1, spot_ID_1, x_position_1, y_position_1)
                                add_a_new_track(spot_ID_2, x_position_2, y_position_2)
                            #spot2 -> track1, spot1 -> new track
                            elif (distance2<=max_move_same_trend) and (distance2<=distance1) and (y_position_2>y1>y2 or y_position_2<y1<y2 ):
                                update_exist_track(track1, spot_ID_2, x_position_2, y_position_2)
                                add_a_new_track(spot_ID_1, x_position_1, y_position_1)
                            else:
                                add_two_new_tracks(spot_ID_1, x_position_1, y_position_1, spot_ID_2, x_position_2, y_position_2)
                        else:
                            raise ValueError("the length of channel_track_spot_dict[channel][track1] is zero!") #??!!
                            
                    #2 cells in the latest frame, 2 spots in the current frame, frame gap <= max gap
                    elif len(channel_latestframe_track[channel])==3: 
                        latestframe, track1, track2 = channel_latestframe_track[channel]
                        max_move_same_trend = same_trend_frame_gap_distance_gap_dict[frame-latestframe]
                        max_move_opposite_trend = opposite_trend_frame_gap_distance_gap_dict[frame-latestframe]
                        x1,y1 = channel_track_spot_dict[channel][track1][-1][-2:]
                        x2,y2 = channel_track_spot_dict[channel][track2][-1][-2:]
                        distance_spot1_1 = abs(y_position_1-y1)
                        distance_spot1_2 = abs(y_position_1-y2)
                        distance_spot2_1 = abs(y_position_2-y1)
                        distance_spot2_2 = abs(y_position_2-y2)
                        #if this is the split event frame, then start two new tracks because it is hard to decide which spot belong to which track
                        if x1==x2 and y1==y2:
                            add_two_new_tracks(spot_ID_1, x_position_1, y_position_1, spot_ID_2, x_position_2, y_position_2)
                        #if all the spot-cell pairs are very distant, then start 2 new tracks
                        elif distance_spot1_1>max_move_opposite_trend and distance_spot1_2>max_move_opposite_trend and distance_spot2_1>max_move_opposite_trend and distance_spot2_2>max_move_opposite_trend: #???!!!!
                            add_two_new_tracks(spot_ID_1, x_position_1, y_position_1, spot_ID_2, x_position_2, y_position_2)
                        else: #at least one cell-spot pair distance is less than max_move_opposite_trend
                            #a Choose Machine
                            #Because distance might be same, so we cannot use this dictionary below!!!
                            # distance_spotinfo_dict = {distance_spot1_1:[spot_ID_1,track1], distance_spot1_2:[spot_ID_1,track2], distance_spot2_1:[spot_ID_2,track1], distance_spot2_2:[spot_ID_2,track2] }
                            spot_ID_xy_dict= {spot_ID_1:[x_position_1, y_position_1], spot_ID_2:[x_position_2, y_position_2] }
                            track1_list = []
                            track2_list = []
                            map_dict = { 1:[spot_ID_1,track1], 2:[spot_ID_1,track2], 3:[spot_ID_2,track1], 4:[spot_ID_2,track2] }
                            distance_index_dict = {1:distance_spot1_1, 2:distance_spot1_2, 3:distance_spot2_1, 4:distance_spot2_2}
                            for key in distance_index_dict:
                                if distance_index_dict[key] <= max_move_opposite_trend:
                                    that_spot_ID, that_track = map_dict[key]
                                    if that_track==track1:
                                        track1_list.append(that_spot_ID)
                                    else:
                                        track2_list.append(that_spot_ID)
                            #if same, then cannot decide which spot is in which track. No!! You Can!!!
                            if set(track1_list) == set(track2_list) and len(track2_list) == 2: #only suitable in this situation, not a general list comparison method
                                #add_two_new_tracks(spot_ID_1, x_position_1, y_position_1, spot_ID_2, x_position_2, y_position_2)
                                if distance_spot1_1 <= distance_spot1_2:
                                    update_exist_track(track1, spot_ID_1, x_position_1, y_position_1)
                                    update_exist_track(track2, spot_ID_2, x_position_2, y_position_2)
                                else:
                                    update_exist_track(track2, spot_ID_1, x_position_1, y_position_1)
                                    update_exist_track(track1, spot_ID_2, x_position_2, y_position_2)
                            elif set(track1_list) == set(track2_list) and len(track2_list) == 1:
                                add_two_new_tracks(spot_ID_1, x_position_1, y_position_1, spot_ID_2, x_position_2, y_position_2)
                                print(f"????????????????????????????please think about this situation {channel} {frame}\n", flush=True)
                            #mannually distributor machine
                            elif len(track1_list) == 0 and len(track2_list) == 1: #1old 1new
                                spot_track_distributor(track2_list[0], track2, spot_ID_xy_dict)
                            elif len(track1_list) == 0 and len(track2_list) == 2: #1old 1new
                                if distance_spot1_2 <= distance_spot2_2: #?? ?= ?
                                    spot_track_distributor(spot_ID_1, track2, spot_ID_xy_dict)
                                else:
                                    spot_track_distributor(spot_ID_2, track2, spot_ID_xy_dict)
                            elif len(track2_list) == 0 and len(track1_list) == 1: #1old 1new
                                spot_track_distributor(track1_list[0], track1, spot_ID_xy_dict)
                            elif len(track2_list) == 0 and len(track1_list) == 2: #1old 1new
                                if distance_spot1_1 <= distance_spot2_1: #?? ?= ?
                                    spot_track_distributor(spot_ID_1, track1, spot_ID_xy_dict)
                                else:
                                    spot_track_distributor(spot_ID_2, track1, spot_ID_xy_dict)
                            elif len(track1_list) == 1: #track2_list could have 2 or 1
                                if len(track2_list) == 2:
                                    track2_list.remove(track1_list[0])
                                update_exist_track(track1, track1_list[0], spot_ID_xy_dict[track1_list[0]][0], spot_ID_xy_dict[track1_list[0]][1])
                                update_exist_track(track2, track2_list[0], spot_ID_xy_dict[track2_list[0]][0], spot_ID_xy_dict[track2_list[0]][1])
                            elif len(track2_list) == 1: #track1_list could have 2 or 1
                                if len(track1_list) == 2:
                                    track1_list.remove(track2_list[0])
                                update_exist_track(track1, track1_list[0], spot_ID_xy_dict[track1_list[0]][0], spot_ID_xy_dict[track1_list[0]][1])
                                update_exist_track(track2, track2_list[0], spot_ID_xy_dict[track2_list[0]][0], spot_ID_xy_dict[track2_list[0]][1])
                            else:
                                raise Exception('Unexpected exception happened when comparing the length to these two lists')
                    #impossible result occurs @.@
                    else:
                        raise Exception('unknown exception happened in 2 spots in the current frame, need further investigation...')
                #3 or over spots in the current frame
                elif len(channel_frame_spot_dict[channel][frame]) >= 3:
                    print("WARNING!!#WARN3  more than 2 spots detected at the same channel and the same frame, please recheck the original input data!")
                    print(f"the channel index is {channel}, the frame is {frame}. Will skip this frame and jump to the next frame\n\n")





###########################################
#Linking Spots - Round Two - link short links to long links, filter fake tracks
###########################################
'''
For Round2 link track segementations first, then filter too short tracks
'''

#some parameters
min_track_frame_duration = 3
max_fake_track_duration = 30 #?
min_track_displacement = 20 #9
max_link_frame_gap=80 #50
max_merge_2track_2track_distance = 30 #20


#some functions
def transient_velocity_calculator(trackID, channel, current_frame, direction, frame_interval=3):
    #frame_interval means need 3 frames to calculate the transient velocity while the time duration is uncertain
    global channel_track_spot_dict
    direction_list = ['forward', 'backward']
    velocity = "False"

    if direction not in direction_list:
        print("Error! Please enter the correct direction (choice: ['forward', 'backward'])")
        exit(0) #!!!??? update the exit syntax
    else:
        spotinfo_list = channel_track_spot_dict[channel][trackID]
        for index in range(len(spotinfo_list)):
            if spotinfo_list[index][0] == current_frame and direction == 'forward':
                if index+frame_interval <= len(spotinfo_list)-1:
                    y_final = spotinfo_list[index+frame_interval][-1]
                    y_current = spotinfo_list[index][-1]
                    time_duration= spotinfo_list[index+frame_interval][0] - current_frame 
                    velocity = (y_final - y_current)/time_duration
                else: #no enough frames left
                    if len(spotinfo_list[index:]) == 1:
                        velocity="SingleSpotTrack"  #this track contains only 1 spot
                    else:
                        y_final = spotinfo_list[-1][-1]
                        y_current = spotinfo_list[index][-1]
                        time_duration= spotinfo_list[-1][0] - current_frame 
                        velocity = (y_final - y_current)/time_duration
                break
            elif spotinfo_list[index][0] == current_frame and direction == 'backward':
                if index >= frame_interval:
                    y_current = spotinfo_list[index][-1]
                    y_previous = spotinfo_list[index-frame_interval][-1]
                    time_duration= current_frame - spotinfo_list[index-frame_interval][0]
                    velocity = (y_current - y_previous)/time_duration
                else:#no enough frames left
                    if index==0:
                        velocity="SingleSpotTrack" #this track contains only 1 spot
                    else:
                        y_current = spotinfo_list[index][-1]
                        y_previous = spotinfo_list[0][-1]
                        time_duration= current_frame - spotinfo_list[0][0]
                        velocity = (y_current - y_previous)/time_duration
                break
    if velocity == "False":
        print("Error! cannot calculate the transient velocity")
        exit(0)
    else:
        if velocity != "SingleSpotTrack":
            velocity = abs(velocity) #????
        if velocity == 0:
            velocity = "SingleSpotTrack" #....!!??
        return velocity

def rearrange_frame_list():
    global channel_track_spot_dict
    for channel in channel_track_spot_dict:
        for track_ID, spot_info_list in channel_track_spot_dict[channel].items():
            frame_list = list()
            frame_dict = dict()
            for frame,spot_ID, x_position, y_position in spot_info_list:
                frame_list.append(frame)
                frame_dict[frame] = [frame,spot_ID, x_position, y_position]
            frame_list = sorted(frame_list)
            new_spot_info_list = list()
            for frame in frame_list:
                new_spot_info_list.append(frame_dict[frame])
            channel_track_spot_dict[channel][track_ID] = new_spot_info_list

def calculate_full_distance(spot_info_list):
    full_distance = 0
    for i in range(len(spot_info_list)-1):
        if spot_info_list[i+1][0] < spot_info_list[i][0]:
            print("wrong!!!!")
            exit(0)
        distance = abs(spot_info_list[i+1][-1] - spot_info_list[i][-1])
        full_distance += distance
    return full_distance
##
##Part 1
##link short tracks to one
##

##Part 1 - link one to another one
if len(spots_data) != 0:
    for channel in channel_filtered_list: 
        #Useful temporary dictionaries
        frame_track_dict = defaultdict(list)  #dict[frame] = [track1,...]
        track_frame_y_dict = defaultdict(dict)  #dict[track][frame] = y_position
        for track_ID in channel_track_spot_dict[channel]:
            for spot_info in channel_track_spot_dict[channel][track_ID]:
                frame = spot_info[0]
                y_position = spot_info[-1]
                frame_track_dict[frame].append(track_ID)
                track_frame_y_dict[track_ID][frame] = y_position
        
        #for one to one round2 linking
        last_track_list_frame = []
        linked_dict = {}
        
        for frame in range (total_frames+1):
            if frame in frame_track_dict:
                current_track_list = frame_track_dict[frame]
                #If the current and previous frame both have only 1 track and these tracks are different, and the frame gap is acceptable
                if (
                    (len(current_track_list)+1 == len(last_track_list_frame)) and 
                    len(current_track_list) == 1 and (frame - last_track_list_frame[-1] <= max_link_frame_gap) and 
                    (set(current_track_list) != set(last_track_list_frame[0:-1]))  
                    ):
                    #compute some basic parameters
                    current_track_ID = current_track_list[0]
                    previous_track_ID = last_track_list_frame[0]
                    if previous_track_ID in linked_dict:
                        previous_track_ID_veryfirst = linked_dict[previous_track_ID]
                    else:
                        previous_track_ID_veryfirst = previous_track_ID

                    y_current = track_frame_y_dict[current_track_ID][frame]
                    y_previous = track_frame_y_dict[previous_track_ID][last_track_list_frame[-1]]
                    current_track_velocity = transient_velocity_calculator(current_track_ID, channel, frame, direction="forward")
                    # print(channel_track_spot_dict[channel].keys())
                    # if previous_track_ID in channel_track_spot_dict[channel]:
                    #     print("Treue")
                    previous_track_velocity = transient_velocity_calculator(previous_track_ID_veryfirst, channel, last_track_list_frame[-1], direction="backward")

                    velocity_similarity = False
                    if current_track_velocity=="SingleSpotTrack" or previous_track_velocity=="SingleSpotTrack":
                        velocity_similarity =True
                    elif abs(current_track_velocity)*0.5 <= abs(previous_track_velocity) <= abs(current_track_velocity)*1.5:  #???!!!!
                        velocity_similarity =True
                    #All the contiditons must be met before these two tracks can be linked: 
                    # 1. a new track 2.displacement short 3.similar transient velocity
                    # if (current_track_ID not in linked_dict) and (abs( y_current-y_previous) <= min_track_displacement) and velocity_similarity==True: #???!!! may miss merge some tracks???

                    if (current_track_ID not in linked_dict) and velocity_similarity==True and \
                          ( (abs( y_current-y_previous) <= min_track_displacement) or (imgHeight*0.3<y_current<imgHeight*0.7 and imgHeight*0.3<y_previous<imgHeight*0.7)  ): #???!!! may miss merge some tracks???
                    # if (current_track_ID not in linked_dict) and (abs( y_current-y_previous) <= min_track_displacement) : #???!!! may miss merge some tracks???
                        #When all the conditions have been met, then check
                        if previous_track_ID not in linked_dict:
                            for spot_info in channel_track_spot_dict[channel][current_track_ID]:
                                channel_track_spot_dict[channel][previous_track_ID].append(spot_info)
                            del channel_track_spot_dict[channel][current_track_ID]
                            linked_dict[current_track_ID] = previous_track_ID
                            linked_dict[previous_track_ID] = 'first one'
                        elif linked_dict[previous_track_ID] == 'first one':
                            for spot_info in channel_track_spot_dict[channel][current_track_ID]:
                                channel_track_spot_dict[channel][previous_track_ID].append(spot_info)
                            del channel_track_spot_dict[channel][current_track_ID]
                            linked_dict[current_track_ID] = previous_track_ID
                        else:
                            very_first_track = linked_dict[previous_track_ID]
                            if linked_dict[very_first_track] != 'first one':
                                print("Unexpected error, how could this track is not the first one?")
                                exit(0)
                            for spot_info in channel_track_spot_dict[channel][current_track_ID]:
                                channel_track_spot_dict[channel][very_first_track].append(spot_info)
                            del channel_track_spot_dict[channel][current_track_ID]
                            linked_dict[current_track_ID] = very_first_track
                #update last_track_list_frame
                last_track_list_frame = current_track_list
                last_track_list_frame.append(frame)
    rearrange_frame_list()

##Part 2 - link two to another two
#Regenerate some dicts again because I changed the channel_frame_spot_dict above
# for channel in channel_frame_spot_dict: 
if len(spots_data) != 0:
    for channel in channel_filtered_list:
        #Useful temporary dictionaries
        frame_track_dict = defaultdict(list)  #dict[frame] = [track1,...]
        track_frame_y_dict = defaultdict(dict)  #dict[track][frame] = y_position
        for track_ID in channel_track_spot_dict[channel]:
            for spot_info in channel_track_spot_dict[channel][track_ID]:
                frame = spot_info[0]
                y_position = spot_info[-1]
                frame_track_dict[frame].append(track_ID)
                track_frame_y_dict[track_ID][frame] = y_position

        #for two to two
        last_track_list_frame = []
        # track_link_dict=defaultdict(list) #the oldest track:linked track
        link_pair_dict = {} #current_track:old track      

        for frame in range (total_frames+1):
            if frame in frame_track_dict:
                current_track_list = frame_track_dict[frame]
                #If the current and previous frame both have 2 tracks and these tracks are different, and the frame gap is acceptable
                if (len(current_track_list)+1 == len(last_track_list_frame)) and len(current_track_list) == 2 and ( (frame - last_track_list_frame[-1]) <= max_link_frame_gap) and (set(current_track_list) != set(last_track_list_frame[0:-1])):
                #consider 2 tracks link to another 2 tracks (trackIDs do not have any overlap)
                #All the contiditons must be met before these four tracks can be linked: 
                # 1. two new tracks 2.displacement short 3.similar transient velocity
                    current_track_ID_1, current_track_ID_2 = current_track_list
                    previous_track_ID_1, previous_track_ID_2 = last_track_list_frame[0:-1]
                    #meet condition No.1
                    if current_track_ID_1 != previous_track_ID_1 and current_track_ID_1 != previous_track_ID_2 and current_track_ID_2 != previous_track_ID_1 and current_track_ID_2 != previous_track_ID_2 :
                        y_current_1 = track_frame_y_dict[current_track_ID_1][frame]
                        y_current_2 = track_frame_y_dict[current_track_ID_2][frame]
                        y_previous_1 = track_frame_y_dict[previous_track_ID_1][last_track_list_frame[-1]]
                        y_previous_2 = track_frame_y_dict[previous_track_ID_2][last_track_list_frame[-1]]
                        distance_11 = abs(y_current_1-y_previous_1)
                        distance_12 = abs(y_current_1-y_previous_2)
                        distance_21 = abs(y_current_2-y_previous_1)
                        distance_22 = abs(y_current_2-y_previous_2)
                        #meet condition No.2  #?????!!!! the condition below only consider the very extreme case -- the two spot - two spot are very close together
                        if distance_11<=max_merge_2track_2track_distance and distance_12<=max_merge_2track_2track_distance and distance_21<=max_merge_2track_2track_distance and distance_22<=max_merge_2track_2track_distance:
                            #check whether the previous track is the first merged track in the track linking part or not
                            if distance_11 <= distance_12:
                                link_pair_dict[current_track_ID_1] = previous_track_ID_1
                                link_pair_dict[current_track_ID_2] = previous_track_ID_2
                            else:
                                link_pair_dict[current_track_ID_1] = previous_track_ID_2
                                link_pair_dict[current_track_ID_2] = previous_track_ID_1
                #update last_track_list_frame
                last_track_list_frame = current_track_list
                last_track_list_frame.append(frame)
        
        #merge two tracks to two tracks, modify the dict
        track_link_chain_dict=defaultdict(list) #the oldest track:linked track
        for key in link_pair_dict:
            #the oldest track
            if link_pair_dict[key] not in link_pair_dict:
                track_link_chain_dict[link_pair_dict[key]].append(key)
            else:
                temp_track = link_pair_dict[key]
                while True:
                    if link_pair_dict[temp_track] not in link_pair_dict:
                        break
                    else:
                        temp_track = link_pair_dict[temp_track]
                track_link_chain_dict[link_pair_dict[temp_track]].append(key)
        for oldest_track,track_list in track_link_chain_dict.items():
            for track in track_list:
                for spot_info in channel_track_spot_dict[channel][track]:
                    # print(channel)
                    # print(oldest_track)
                    # print(channel_track_spot_dict[channel].keys())
                    channel_track_spot_dict[channel][oldest_track].append(spot_info)
                print('find one')
                del channel_track_spot_dict[channel][track]
rearrange_frame_list()


##
## force gap closing for motionless track #one to one
##
channel_frame_list_dict = defaultdict(list)
channel_frame_info_dict = defaultdict(dict)
for channel in channel_filtered_list:
    for track_ID in channel_track_spot_dict[channel]:
        for frame,spot_ID, x_position, y_position  in channel_track_spot_dict[channel][track_ID]:
            if frame in channel_frame_info_dict[channel]:
                channel_frame_info_dict[channel][frame].append([track_ID, y_position])
            else:
                channel_frame_info_dict[channel][frame] = list()
                channel_frame_info_dict[channel][frame].append([track_ID, y_position])
            channel_frame_list_dict[channel].append(frame)

channel_force_gap_close_dict = defaultdict(dict)
for channel in channel_filtered_list:
    force_gap_closing_dict = dict()
    frame_list = channel_frame_list_dict[channel]
    frame_list = sorted(frame_list)
    for i in range(len(frame_list)-1):
        current_frame = frame_list[i]
        next_frame = frame_list[i+1]
        #one to one
        if (len(channel_frame_info_dict[channel][current_frame]) == 1) and (len(channel_frame_info_dict[channel][next_frame]) == 1):
            current_track_ID, current_y_position = channel_frame_info_dict[channel][current_frame][0]
            next_track_ID, next_y_position = channel_frame_info_dict[channel][next_frame][0]
            if current_track_ID != next_track_ID:
                # print('===')
                # print(current_track_ID)
                # print(next_track_ID)
                # print('\n')
                if (frame_list[i+1] - frame_list[i]) >max_link_frame_gap and abs(current_y_position-next_y_position)< imgHeight*0.05: 
                    force_gap_closing_dict[next_track_ID] = current_track_ID
                elif max_link_frame_gap< (frame_list[i+1] - frame_list[i]) < max_link_frame_gap*2 \
                    and imgHeight*0.2<=current_y_position<=imgHeight*0.8 and imgHeight*0.2<=next_y_position<=imgHeight*0.8 \
                    and abs(current_y_position-next_y_position)< imgHeight*0.2:  #?????!!!
                    force_gap_closing_dict[next_track_ID] = current_track_ID
                elif (frame_list[i+1] - frame_list[i]) <= max_link_frame_gap and abs(current_y_position-next_y_position)< imgHeight*0.3 \
                    and imgHeight*0.2<=current_y_position<=imgHeight*0.8 and imgHeight*0.2<=next_y_position<=imgHeight*0.8 : #?????!!!
                    force_gap_closing_dict[next_track_ID] = current_track_ID

    force_gap_close_dict = defaultdict(list)
    for next_track_ID, current_track_ID in force_gap_closing_dict.items():
        while True:
            if current_track_ID not in force_gap_closing_dict:
                force_gap_close_dict[current_track_ID].append(next_track_ID)
                break
            else:
                current_track_ID =force_gap_closing_dict[current_track_ID]
    for track_ID, track_list in force_gap_close_dict.items():
        channel_force_gap_close_dict[channel][track_ID] = track_list


for channel in channel_filtered_list:
    if channel in channel_force_gap_close_dict:
        for track_ID, track_list in channel_force_gap_close_dict[channel].items():
            for track_del in track_list:
                for spot_info in channel_track_spot_dict[channel][track_del]:
                    channel_track_spot_dict[channel][track_ID].append(spot_info)
                del channel_track_spot_dict[channel][track_del]
rearrange_frame_list()






##
##Part 2
##filter too short, nearly zero displacement, too close to the bottom or top of the channel
##
del_track_list = []
# for channel in channel_frame_spot_dict: 
if len(spots_data) != 0:
    for channel in channel_filtered_list:
        for track_ID in channel_track_spot_dict[channel]:
            #tooo short 1frame duration
            if len(channel_track_spot_dict[channel][track_ID]) <= 2:
                # pass
                del_track_list.append([channel,track_ID])
            else:
                frame_duration = channel_track_spot_dict[channel][track_ID][-1][0] - channel_track_spot_dict[channel][track_ID][0][0]
                displacement = abs(channel_track_spot_dict[channel][track_ID][-1][-1] - channel_track_spot_dict[channel][track_ID][0][-1])
                final_y_position = channel_track_spot_dict[channel][track_ID][-1][-1]
                full_distance = calculate_full_distance(channel_track_spot_dict[channel][track_ID])
                somekind_velocity = displacement/frame_duration
                y_position_list = [i[-1] for i in channel_track_spot_dict[channel][track_ID]]
                extremum = max(y_position_list)-min(y_position_list)
                #short distance
                if full_distance <=imgHeight*0.2: #???!!!
                    del_track_list.append([channel,track_ID])
                #wandering around a small area
                elif extremum<= min_track_displacement*2 :
                     del_track_list.append([channel,track_ID])
                #not at the bottom/top of the channel, too short frame duration, too short displacement
                elif frame_duration<=max_fake_track_duration and displacement <= min_track_displacement and final_y_position>=imgHeight*0.1 \
                                                and final_y_position<=imgHeight*0.9 and channel_track_spot_dict[channel][track_ID][0][0]< total_frames*0.8: #??
                    del_track_list.append([channel,track_ID])
                #at the bottom/top of the channel, too short frame duration, too short displacement #??!!
                elif frame_duration<=(max_fake_track_duration*2) and  displacement <= min_track_displacement and (final_y_position<=imgHeight*0.1 or final_y_position>=imgHeight*0.9 ):
                    del_track_list.append([channel,track_ID])
                #short displacement and not at the end of the vedio
                elif full_distance <= imgHeight*0.8 and channel_track_spot_dict[channel][track_ID][-1][0]< (total_frames*0.95) and somekind_velocity<40:
                    del_track_list.append([channel,track_ID])
    for channel, track_ID in del_track_list:
        del channel_track_spot_dict[channel][track_ID]







###########################################
#Output Save
###########################################
update_total_tracks = 0
with open (args.linking_output, 'w') as output_file:
    output_file.write('\t'.join(["SPOT_ID", "CHANNEL_ID", "TRACK_ID", "FRAME", "POSITION_X", "POSITION_Y" ])+'\n')
    # for channel in channel_frame_spot_dict: 
    for channel in channel_filtered_list:
        for track_ID in channel_track_spot_dict[channel]: #to avoid re-assign the track_index variable
            update_total_tracks+=1
            for spot_metadata in channel_track_spot_dict[channel][track_ID]:
                frame,spot_ID, x_position, y_position = spot_metadata
                output_file.write('\t'.join([str(spot_ID), str(channel), str(track_ID), str(frame), str(x_position), str(y_position)])+'\n')



print("\nThe total number of tracks is: "+str(update_total_tracks))







###########################################
#Track Classification
###########################################
fast_slow_time_threshold = 25
fast_distance_cutoff = imgHeight*0.6
channel_track_classification_dict = defaultdict(dict)
classification_track_dict = dict()
classification_track_dict["up-fast"] = list()
classification_track_dict["up-slow"] = list()
classification_track_dict["down-fast"] = list()
classification_track_dict["down-slow"] = list()
classification_track_dict["zero"] = list()

for channel in channel_filtered_list:
    for track_ID in channel_track_spot_dict[channel]:
        frame_duration = channel_track_spot_dict[channel][track_ID][-1][0] - channel_track_spot_dict[channel][track_ID][0][0]
        total_distance = calculate_full_distance(channel_track_spot_dict[channel][track_ID])
        net_displacement = channel_track_spot_dict[channel][track_ID][-1][-1] - channel_track_spot_dict[channel][track_ID][0][-1]
        if net_displacement > 0: #UP
            if frame_duration <= fast_slow_time_threshold and total_distance >= fast_distance_cutoff:
                channel_track_classification_dict[channel][track_ID] = "up-fast"
                classification_track_dict["up-fast"].append([channel, track_ID])
            else:
                channel_track_classification_dict[channel][track_ID] = "up-slow"
                classification_track_dict["up-slow"].append([channel, track_ID])
        elif net_displacement <0 :
            if frame_duration <= fast_slow_time_threshold and total_distance >= fast_distance_cutoff:
                channel_track_classification_dict[channel][track_ID] = "down-fast"
                classification_track_dict["down-fast"].append([channel, track_ID])
            else:
                channel_track_classification_dict[channel][track_ID] ="down-slow"
                classification_track_dict["down-slow"].append([channel, track_ID])
        else: #net_displacement ==0 
            channel_track_classification_dict[channel][track_ID] ="zero"
            classification_track_dict["zero"].append([channel, track_ID])

classification_result = list()
for type in ["up-fast", "up-slow","down-fast","down-slow", "zero"]:
    classification_result.append(len(classification_track_dict[type]))



# choose the specific type of move mode classification to record
# for channel in channel_track_classification_dict:
#     for track_ID in channel_track_classification_dict[channel]:
#         if channel_track_classification_dict[channel][track_ID] !="up-slow": #the one you want to keep
#             del channel_track_spot_dict[channel][track_ID]
# classification_result = [0]*5
# classification_result[1] = len(classification_track_dict["up-slow"])





###########################################
#Basic Figures!!!!
###########################################
stop_move = 5 #the max value that can be considered as not move...
total_tracks = update_total_tracks
track_velocity_dict = defaultdict(dict)
straight_line_velocity_list = list()
straight_line_velocity_up_list = list()
straight_line_velocity_down_list = list()

fig = plt.figure(figsize=(16,9))
gs = fig.add_gridspec(3,3, hspace=0.4, left=0.08, right=0.95, bottom=0.05)

if args.plot_out:
    plt.suptitle(args.plot_out)
else:
    plt.suptitle(args.image_folder.replace("D:\\Trajectory_Project_Channel\\", "").replace("\\","").replace("_Channel_only",""))




#Channel basic info bar plot
###########################################
ax_info = fig.add_subplot(gs[0, 0])
ax_info.set_title("Channel Basic Info")
ax_info.bar(x=("Total", "Has Signals", "Max One", "Max Two"), 
        height=(len(channel_features),  channel_number_atleast_has_spot, channel_max_one, channel_max_two), alpha=0.3)
for i, v in enumerate([len(channel_features),  channel_number_atleast_has_spot, channel_max_one, channel_max_two]):
    ax_info.text(y=v-3, x=i, s=str(v), color='black', fontweight='normal', horizontalalignment='center')

ax_info.set_ylim([0,70])
ax_info.set_ylabel("Number")



#Turnover and Stop Ratio Barplot
###########################################
#calculate each ratio
turnover_ratio_list = list()
stop_ratio_list = list()
for channel in channel_filtered_list:
    for track_ID in channel_track_spot_dict[channel]:
        #some parameters
        single_track_turnover_count = 0
        single_track_stop_count = 0
        current_trend = 'up'
        complement_frames_list = list()
        #basic info
        frame_duration = channel_track_spot_dict[channel][track_ID][-1][0] - channel_track_spot_dict[channel][track_ID][0][0]
        start_frame = channel_track_spot_dict[channel][track_ID][0][0]
        end_frame = channel_track_spot_dict[channel][track_ID][-1][0]
        #build dictionary
        track_frame_y_dict = dict()
        for track_info_list in channel_track_spot_dict[channel][track_ID]:
            track_frame_y_dict[track_info_list[0]] = track_info_list[-1]
        for frame in range(start_frame, end_frame): #assume that the cell does not move when no record
            if frame not in track_frame_y_dict:
                complement_frames_list.append(frame)
                previous_frame = frame - 1
                while True:
                    if previous_frame not in track_frame_y_dict:
                        previous_frame = previous_frame -1
                    else:
                        break
                track_frame_y_dict[frame] = track_frame_y_dict[previous_frame]

        for i in range(start_frame+1, end_frame):
            past, current, next_ = (i-1, i, i+1)
            past_y = track_frame_y_dict[i-1]
            current_y = track_frame_y_dict[i]
            next_y = track_frame_y_dict[i+1]
            #turnover possibility
            if abs(next_y - current_y) > stop_move: 
                if (current_y - past_y) * (next_y - current_y) < 0 :
                    single_track_turnover_count += 1
                elif current_y - past_y == 0: 
                    continued_stop_start_frame = i-1
                    while True:
                        if continued_stop_start_frame-1 in track_frame_y_dict:
                            if track_frame_y_dict[continued_stop_start_frame] == track_frame_y_dict[continued_stop_start_frame -1]:
                                continued_stop_start_frame -= 1
                            else:
                                break
                        else:
                            break
                    if continued_stop_start_frame-1 in track_frame_y_dict:
                        if (track_frame_y_dict[continued_stop_start_frame] - track_frame_y_dict[continued_stop_start_frame-1]) * (next_y - current_y) <0:
                            single_track_turnover_count += 1
            #stop possibility
            else: 
                if i not in complement_frames_list:
                    if (current_y - past_y) * (next_y - current_y) <= 0 :
                        single_track_stop_count += 1
                    else:
                        if abs(next_y - past_y) <= stop_move:
                            single_track_stop_count += 1
        
        #Compute ratios
        stop_ratio = single_track_stop_count/(frame_duration-len(complement_frames_list)-1)
        turnover_ratio = single_track_turnover_count/(frame_duration-len(complement_frames_list)-1)
        turnover_ratio_list.append(turnover_ratio)
        stop_ratio_list.append(stop_ratio)


if channel_number_atleast_has_spot != 0:
    height_list = [(channel_max_one+channel_max_two)/channel_number_atleast_has_spot, np.mean(turnover_ratio_list), np.mean(stop_ratio_list)]
else:
    height_list = [0, np.mean(turnover_ratio_list), np.mean(stop_ratio_list)]
erro_list = [0,  np.std(turnover_ratio_list), np.std(stop_ratio_list)]
ax_ratios = fig.add_subplot(gs[0, 1])
ax_ratios.set_title("Some Ratios")
ax_ratios.set_ylabel("Ratio")
ax_ratios.set_ylim([0, 1])
ax_ratios.bar(x=("Used Channel Ratio", "Turnover Ratio", "Stop Ratio"), 
        height=height_list, yerr=erro_list ,align='center', width=0.5, alpha=0.5, color=('green','orange', 'green'))
for i, v in enumerate(height_list):
    ax_ratios.text(y=v, x=i, s=str(round(v,3)), color='black', fontweight='normal', horizontalalignment='center')



# for i, v in enumerate([len(channel_features),  channel_number_atleast_has_spot, channel_max_one, channel_max_two]):
#     ax_info.text(y=v-3, x=i, s=str(v), color='black', fontweight='normal', horizontalalignment='center')



#All other indices
###########################################
all_instant_speed_list = list()
lines = []
colors = []
my_cmap = plt.get_cmap('jet')

total_distance_list = []
net_displacement_list = []
time_duration_list = []
mean_curvilinear_speed_list = []
mean_straight_line_speed_list = []
hesitate_index_list = []
hesitate_index_list_1track1value = []
hesitiation_time_ratio_list = []


for channel in channel_filtered_list:
    for track_ID in channel_track_spot_dict[channel]:
        #some parameters
        single_track_instant_speed_list = list()
        total_distance = 0

        #basic info
        frame_duration = channel_track_spot_dict[channel][track_ID][-1][0] - channel_track_spot_dict[channel][track_ID][0][0]
        start_frame = channel_track_spot_dict[channel][track_ID][0][0]
        end_frame = channel_track_spot_dict[channel][track_ID][-1][0]
        net_displacement = channel_track_spot_dict[channel][track_ID][-1][-1] - channel_track_spot_dict[channel][track_ID][0][-1]
        
        #net_displacement, time duration, straight line speed
        net_displacement_list.append(net_displacement)
        time_duration_list.append(frame_duration)
        mean_straight_line_speed_list.append(net_displacement/frame_duration)

        #instantaneous velocity and distance
        track_record_list = channel_track_spot_dict[channel][track_ID]
        for i in range(len(track_record_list)-1):
            first, first_y = track_record_list[i][0], track_record_list[i][-1]
            second, second_y = track_record_list[i+1][0], track_record_list[i+1][-1]
            displacement = second_y - first_y
            #distance
            total_distance += abs(displacement)
            #velocity
            if second -first <= 3:
                velocity = displacement/(second -first)
                single_track_instant_speed_list.append(velocity)
                for i in range(second -first):
                    all_instant_speed_list.append(velocity)
        
        #hesitation
        track_segs_list = list()
        single_seg = list()
        single_seg.append([track_record_list[0][0], track_record_list[0][-1]])
        for i in range(len(track_record_list)-1):
            first, first_y = track_record_list[i][0], track_record_list[i][-1]
            second, second_y = track_record_list[i+1][0], track_record_list[i+1][-1]
            if second -first <= 3: #???!!!
                single_seg.append([second, second_y])
            else:
                if len(single_seg) >= 3: #???!!!
                    track_segs_list.append(single_seg)
                single_seg=list()
                single_seg.append([second, second_y])
        if len(single_seg) >= 3:
            track_segs_list.append(single_seg)
        for single_seg in track_segs_list:
            displacement = abs(single_seg[-1][-1] - single_seg[0][-1])
            distance = 0
            for index in range(len(single_seg)-1):
                distance+= abs(single_seg[index+1][-1] - single_seg[index][-1])
            if distance != 0:
                hesitate_index_list.append(displacement/distance)
            # if displacement != 0:
            #     hesitate_index_list.append(distance/displacement) #??!!! maybe we should use one index for one track not several number for one track
            # else:
            #     hesitate_index_list.append(distance/0.1)

        used_frame =0
        for i in track_segs_list:
            used_frame+= len(i)
        hesitiation_time_ratio_list.append(used_frame/(frame_duration+1))
        #heisitation index 1 track 1value
        hesitate_index_list_1track1value.append(abs(net_displacement/total_distance))
        # if net_displacement == 0:
        #     hesitate_index_list_1track1value.append(total_distance/0.1)
        # else:
        #     hesitate_index_list_1track1value.append(abs(total_distance/net_displacement))
        
        #for multiple line chart
        lines.append( list(enumerate(single_track_instant_speed_list) )  )
        colors.append(my_cmap(np.random.rand()))

        #for other plots
        total_distance_list.append(total_distance)
        curvilinear_speed_list = [ abs(i) for i in single_track_instant_speed_list]
        mean_curvilinear_speed_list.append(np.mean(curvilinear_speed_list))




#Instantaneous Speed multiple line chart
########################################### 
# for i in lines:
#     print(i)
print(len(lines))
ax_speedlines = fig.add_subplot(gs[0, 2])
ax_speedlines.set_title("Instantaneous Speed of Each Track")
line_segments = LineCollection(lines, linewidth=0.5,colors=colors, linestyle='solid')
ax_speedlines.set_xlabel("Time Duration (second)")
ax_speedlines.set_ylabel("Speed (pixel/second)")
ax_speedlines.set_xlim([0,1000])
ax_speedlines.set_ylim([-30,50])
ax_speedlines.add_collection(line_segments)



#total distance histogram
###########################################
ax_total_Distance = fig.add_subplot(gs[1,0])
ax_total_Distance.set_title("Track Total Distance Distribution")
if len(total_distance_list) >0:
    ax_total_Distance.hist(total_distance_list, bins=np.arange(0, max(total_distance_list), 10), log=True, alpha=0.8)
else:
    ax_total_Distance.hist(total_distance_list, bins=np.arange(0, 100, 10), log=True, alpha=0.8)

ax_total_Distance.set_xlabel("distance (pixel)")
ax_total_Distance.set_ylabel("log number of tracks")
# ax_total_Distance.set_xlim([-100, 100])
# ax_total_Distance.set_ylim([0, 50])

#Net displacement histogram
###########################################
ax_net_disp = fig.add_subplot(gs[1,1])
ax_net_disp.set_title("Track Net Displacement Distribution")
if len(net_displacement_list)>0:
    ax_net_disp.hist(net_displacement_list, bins=np.arange(min(net_displacement_list), max(net_displacement_list), 10), log=True, alpha=0.8)
else:
    ax_net_disp.hist(net_displacement_list, bins=np.arange(-50, 50, 10), log=True, alpha=0.8)
ax_net_disp.set_xlabel("displacement (pixel)")
ax_net_disp.set_ylabel("log number of tracks")
# ax_net_disp.set_xlim([-100, 100])
# ax_net_disp.set_ylim([0, 50])




#Time duration histogram
###########################################
ax_time= fig.add_subplot(gs[1,2])
ax_time.set_title("Track Time Duration Distribution")
if len(time_duration_list) >0:
    ax_time.hist(time_duration_list, bins=np.arange(0, max(time_duration_list), 50), log=True, alpha=0.8)
else:
    ax_time.hist(time_duration_list, bins=np.arange(0, 3000, 50), log=True, alpha=0.8)
ax_time.set_xlabel("time duration(second)")
ax_time.set_ylabel("log number of tracks")
# ax_time.set_xlim([-100, 100])
# ax_time.set_ylim([0, 50])




#Instantaneous Speed Histogram
###########################################
if len(all_instant_speed_list) >0:
    if min(all_instant_speed_list) < -130 or max(all_instant_speed_list) >150:
        print("Warning !!!!!!!!!!! Exceed the threshold")
        print("Warning !!!!!!!!!!! Exceed the threshold")
        print("Warning !!!!!!!!!!! Exceed the threshold")
        print("Warning !!!!!!!!!!! Exceed the threshold")
        print("Warning !!!!!!!!!!! Exceed the threshold")
        print(min(all_instant_speed_list))
        print(max(all_instant_speed_list))
        exit(0)
ax_instantaneous_hist = fig.add_subplot(gs[2, 0])
ax_instantaneous_hist.set_title("Instantaneous Speed Histogram")
ax_instantaneous_hist.hist(all_instant_speed_list, bins=np.arange(-120, 150, 2), log=True)
ax_instantaneous_hist.set_xlabel("Displacement (pixel)")
ax_instantaneous_hist.set_ylabel("log number of displacement")
ax_instantaneous_hist.set_xlim([-120, 150])
# ax_instantaneous_hist.set_ylim([])


#Mean Curvilinear speed Histogram
###########################################
ax_mean_cur= fig.add_subplot(gs[2,1])
ax_mean_cur.set_title("Mean Curvilinear Speed Distribution")
if len(mean_curvilinear_speed_list) >0:
    ax_mean_cur.hist(mean_curvilinear_speed_list, bins=np.arange(min(mean_curvilinear_speed_list), max(mean_curvilinear_speed_list), 2), log=True, alpha=0.8)
else:
    ax_mean_cur.hist(mean_curvilinear_speed_list, bins=np.arange(0, 100, 2), log=True, alpha=0.8)
ax_mean_cur.set_xlabel("speed (pixel/second)")
ax_mean_cur.set_ylabel("log number of tracks")
# ax_mean_cur.set_xlim([-100, 100])
# ax_mean_cur.set_ylim([0, 50])





#Mean Straight-line Speed histogram
###########################################
ax_slSpeed = fig.add_subplot(gs[2,2])
ax_slSpeed.set_title("Mean Straight-line Speed Distribution")

if len(mean_straight_line_speed_list) >0 :
    if max(mean_straight_line_speed_list) > 100 or min(mean_straight_line_speed_list) <-100:
        print("Warning! The max or min velocity is out of range!")
        print("Warning! The max or min velocity is out of range!")
        print("Warning! The max or min velocity is out of range!")
        print("Warning! The max or min velocity is out of range!")
        print("Warning! The max or min velocity is out of range!")
        print(max(mean_straight_line_speed_list))
        print(min(mean_straight_line_speed_list))
        exit(0)

ax_slSpeed.hist(mean_straight_line_speed_list, bins=np.arange(-100, 100, 2), log=True)
ax_slSpeed.set_xlabel("velocity (pixel/second)")
ax_slSpeed.set_ylabel("log number of tracks")
ax_slSpeed.set_xlim([-100, 100])
# ax_slSpeed.text(x=0.5, y=0.5, transform = ax_slSpeed.transAxes,  s=f"total tracks: {total_tracks}\ntrack with positive velocity: {len(straight_line_velocity_up_list)}\ntrack with negative velocity: {len(straight_line_velocity_down_list)}")







if args.plot_out :
    plot_folder = "C:/Users/cotton/OneDrive/XF_Lab/Trajectory/Channel_Velocity/Tracking_Result/"
    plt.savefig(plot_folder+args.plot_out+'.jpg',dpi=300)



    with open("C:/Users/cotton/OneDrive/XF_Lab/Trajectory/Channel_Velocity/Experiment_Ratio_result.txt",'a') as file_object:
        file_object.seek(0)
        BigCategory, experiment_key_word = args.plot_out.split('__')
        file_index_dict = {1:"total_distance", 2:"net_displacement", 3:"time_duration", 4:"instantaneous_speed", 5:"mean_curvilinear_speed", 6:"mean_straight_line_speed", 
              7:"turnover_ratio", 8:"stop_ratio", 9:"cell_number", 10:"hesitation_index", 11:"hesitation_used_time_ratio", 12:"track_classification", 13:"channel_usage"}
        for i, index in file_index_dict.items():
            if i==1:
                mylist = total_distance_list
            elif i==2:
                mylist = net_displacement_list
            elif i==3:
                mylist = time_duration_list
            elif i==4:
                mylist = all_instant_speed_list
            elif i==5:
                mylist = mean_curvilinear_speed_list
            elif i==6:
                mylist = mean_straight_line_speed_list
            elif i==7:
                mylist = turnover_ratio_list
            elif i==8:
                mylist = stop_ratio_list
            elif i==9:
                mylist = [update_total_tracks]
            elif i==10:
                # mylist = hesitate_index_list_1track1value #?????!!!
                mylist = hesitate_index_list
            elif i==11:
                mylist = hesitiation_time_ratio_list
            elif i==12:
                mylist = classification_result
            elif i==13:
                if channel_number_atleast_has_spot == 0:
                    channel_usage_ratio = 0
                else:
                    channel_usage_ratio = (channel_max_one+channel_max_two)/channel_number_atleast_has_spot
                mylist = [channel_usage_ratio,channel_number_atleast_has_spot ]

            
            file_object.write(BigCategory+'\t'+experiment_key_word+'\t'+str(i)+'\t'+'\t'.join( [ str(i) for i in mylist])  +'\n')

else:
    plt.show()



