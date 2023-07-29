# MIT License 
# Author: winstonyym

import requests
import json
import torch
import glob
import os
import numpy as np
import shutil
import argparse
import logging
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
import warnings
import time
from tqdm import tqdm
from collections import Counter

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 65
CLS_DICT = {'0': 'Bird',
 '1': 'Ground-Animal',
 '2': 'Curb',
 '3': 'Fence',
 '4': 'Guard-Rail',
 '5': 'Barrier',
 '6': 'Wall',
 '7': 'Bike-Lane',
 '8': 'Crosswalk---Plain',
 '9': 'Curb-Cut',
 '10': 'Parking',
 '11': 'Pedestrian-Area',
 '12': 'Rail-Track',
 '13': 'Road',
 '14': 'Service-Lane',
 '15': 'Sidewalk',
 '16': 'Bridge',
 '17': 'Building',
 '18': 'Tunnel',
 '19': 'Person',
 '20': 'Bicyclist',
 '21': 'Motorcyclist',
 '22': 'Other-Rider',
 '23': 'Lane-Marking---Crosswalk',
 '24': 'Lane-Marking---General',
 '25': 'Mountain',
 '26': 'Sand',
 '27': 'Sky',
 '28': 'Snow',
 '29': 'Terrain',
 '30': 'Vegetation',
 '31': 'Water',
 '32': 'Banner',
 '33': 'Bench',
 '34': 'Bike-Rack',
 '35': 'Billboard',
 '36': 'Catch-Basin',
 '37': 'CCTV-Camera',
 '38': 'Fire-Hydrant',
 '39': 'Junction-Box',
 '40': 'Mailbox',
 '41': 'Manhole',
 '42': 'Phone-Booth',
 '43': 'Pothole',
 '44': 'Street-Light',
 '45': 'Pole',
 '46': 'Traffic-Sign-Frame',
 '47': 'Utility-Pole',
 '48': 'Traffic-Light',
 '49': 'Traffic-Sign-(Back)',
 '50': 'Traffic-Sign-(Front)',
 '51': 'Trash-Can',
 '52': 'Bicycle',
 '53': 'Boat',
 '54': 'Bus',
 '55': 'Car',
 '56': 'Caravan',
 '57': 'Motorcycle',
 '58': 'On-Rails',
 '59': 'Other-Vehicle',
 '60': 'Trailer',
 '61': 'Truck',
 '62': 'Wheeled-Slow',
 '63': 'Car-Mount',
 '64': 'Ego-Vehicle'}

# Get helper function
def addIndice(output_max):
    set_of_pixels = torch.unique(output_max, return_counts=True)
    set_dictionary = {}
    for i in range(NUM_CLASSES):
            set_dictionary[str(i)] = 0
    for pixel,count in zip(set_of_pixels[0], set_of_pixels[1]):
        set_dictionary[str(pixel.item())] = count.item()
    set_dictionary['Total'] = int(np.sum(list(set_dictionary.values())))
    return set_dictionary

def addInstance(output_max):
    list_unique, list_counts = torch.unique(output_max[0]['segmentation'].int(), return_counts=True)

    if -1 in list_unique:
        list_unique = list_unique[1:]
        list_counts = list_counts[1:]

    total = torch.sum(list_counts).item()

    matching_dict = {}
    for i, k in zip(range(len(output_max[0]['segments_info'])), output_max[0]['segments_info']):
        matching_dict[i] = int(k['label_id'])

    set_dictionary = {}
    for i in range(NUM_CLASSES):
                set_dictionary[str(i)] = 0

    for i, k in zip(list_unique, list_counts):
        set_dictionary[str(matching_dict[i.item()])] += k.item()
        
    set_dictionary['Total'] = total

    return set_dictionary

def addInstanceCounts(output_max):

    instance_dictionary = {}
    
    instance_dictionary = {}
    for i in range(NUM_CLASSES):
                instance_dictionary[str(i)] = 0
    
    # for each segment, draw its legend
    for segment in output_max[0]['segments_info']:
        segment_id = segment['id']
        segment_label_id = str(segment['label_id'])
        instance_dictionary[segment_label_id] += 1

    return instance_dictionary

# Load Mask2Former
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-panoptic")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-mapillary-vistas-panoptic")
model = model.to(device)

# Configure logger
logging.basicConfig(filename="std.log", format='%(asctime)s %(message)s', filemode='w') 
logger=logging.getLogger() 
logger.setLevel(logging.INFO) 

# Setup arguments
parser = argparse.ArgumentParser(description="Main module that runs segmentation workflow on folder")

# Add positional arguments
parser.add_argument('input_path', help='Filepath to image data', type=str)
parser.add_argument('threshold', default=1.0, help='Image segmentation threshold value', type=float)

args = parser.parse_args()

def main():

    # Set start state
    start = 0
    image_indicators_dict = {}
    image_instances_dict = {}

    # Create output folder if none exist
    if not os.path.exists(f'./instance_outputs'):
        os.makedirs(f'./instance_outputs')

    # Get list of images
    image_set = [i for i in os.listdir(os.path.join(os.getcwd(),args.input_path))]

    for i, image in enumerate(tqdm(image_set)):
        
        print(f'Segmenting {image}')
        img = Image.open(os.path.join(os.getcwd(),args.input_path, f'{image}'))
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            pixel_values = inputs['pixel_values'].to(device)
            pixel_mask = inputs['pixel_mask'].to(device)
            outputs = model(pixel_values = pixel_values, pixel_mask = pixel_mask)
        out = processor.post_process_instance_segmentation(outputs, target_sizes=[img.size[::-1]], threshold=args.threshold)
        image_indicators_dict[image] = addInstance(out)
        image_instances_dict[image] = addInstanceCounts(out)
        logger.info(f"Segmented {i}:{image}")

    
        if i != 0 and i % 10000 == 0:
            df = pd.DataFrame.from_dict(image_indicators_dict, orient='index')
            df = df.rename(mapper = CLS_DICT, axis=1)
            df2 = pd.DataFrame.from_dict(image_instances_dict, orient='index')
            df2 = df2.rename(mapper = CLS_DICT, axis=1)
            df.to_csv(os.path.join(os.getcwd(),f'instance_outputs/{start}-{i}_output.csv'), index = True, header=True)
            df2.to_csv(os.path.join(os.getcwd(),f'instance_outputs/{start}-{i}_instances.csv'), index = True, header=True)
            start = i
            image_indicators_dict = {}
            image_instances_dict = {}
            logger.info(f"Saved CSV checkpoint for images {start}:{i}")
        
        
    df = pd.DataFrame.from_dict(image_indicators_dict, orient='index')
    df = df.rename(mapper = CLS_DICT, axis=1)
    df.to_csv(os.path.join(os.getcwd(),f'instance_outputs/{start}-{len(image_set)}_output.csv'), index = True, header=True)
    
    df2 = pd.DataFrame.from_dict(image_instances_dict, orient='index')
    df2 = df2.rename(mapper = CLS_DICT, axis=1)
    df2.to_csv(os.path.join(os.getcwd(),f'instance_outputs/{start}-{len(image_set)}_instances.csv'), index = True, header=True)
    logger.info(f"Saved CSV checkpoint for images {start}:{len(image_set)}")
        
        
if __name__ == "__main__":
   main()
   
