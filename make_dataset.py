'''
    Dataset of IMAGE_SIZE x IMAGE_SIZE images will be created.
    This dataset will be saved to PICKLE_PATH/PREFIX_STR.#.pickle

    To use:
        Set params in the User configurables area
    To run:
        python make_dataset.py
'''
import tensorflow as tf 
import numpy as np
import argparse
import os
import sys
import random
import itertools
from six.moves import cPickle as pickle
from collections import Counter
from PIL import Image
from PIL import ImageEnhance
import PIL.ImageOps

def get_key_virgin_islands(filename):
    return int(filename[3:-4])
def get_key_panama(filename):
    return int(filename[21:-4])    

root_dir = os.environ['DATA_HOME'];

# ---------------------- User configurables -------------------------------#
USER_KEY = get_key_panama # Define function for sorting data chronologically
# ---------------------- User input -------------------------------#

'''
Load the images for a single sound class
ACCEPTS: a sound label, that must match one of the folder names in data/bins/whistles
'''
def load_files(filenames, index, dp): 
    if index < len(filenames) / dp['FILE_CHUNK']:
        image_files = filenames[index*dp['FILE_CHUNK']:(index+1)*dp['FILE_CHUNK']]
    else:
        image_files = filenames[index*dp['FILE_CHUNK']:]

    dataset = np.ndarray(shape=(len(image_files), dp['IMAGE_SIZE'], dp['IMAGE_SIZE'], dp['NUM_CHANNELS']),
                            dtype=np.float32)

    filenames =  []

    num_images = 0
    for image in image_files: 
        image_file = os.path.join(dp['DATA_PATH'], image) 
        try:
            # Read_image_from_file should already preform downsampling
            print "Reading", image_file 
            image_data = read_image_from_file(image_file, dp) 
            dataset[num_images, :, :, :] = image_data 
            num_images = num_images + 1

        except IOError as e:
            print('Could not read:', image_file, ':', e)

    dataset = dataset[0:num_images, :, :, :] 
    print 'Dataset size:', dataset.shape
    print 'Mean:', np.mean(dataset) 
    print 'Standard deviation:', np.std(dataset)
    print '' 
    return dataset

def read_image_from_file(file_path, dp):
    img = Image.open(file_path)
    img = img.resize((dp['IMAGE_SIZE'],dp['IMAGE_SIZE']), Image.ANTIALIAS) #downsample image
    pixel_values = np.array(img.getdata())
    return np.reshape(pixel_values, [dp['IMAGE_SIZE'], dp['IMAGE_SIZE'], dp['NUM_CHANNELS']])

def scale_pixel_values(dataset, dp):
    return (dataset - dp['PIXEL_DEPTH'] / 2.0) / dp['PIXEL_DEPTH']

def make_basic_datasets(dp):
    image_path = dp['DATA_PATH']
    files = [x for x in os.listdir(image_path)]
    files.sort(key = USER_KEY)
    print len(files), "files found."

    for index in xrange(len(files)/dp['FILE_CHUNK']+ 1):
        # Downsample and load files
        print "Loading images from index", index*dp['FILE_CHUNK'], "to", (index+1)*dp['FILE_CHUNK']
        data = load_files(files, index, dp)
        data = scale_pixel_values(data, dp)

        # save all the datasets in a pickle file
        pickle_file =  dp['PREFIX_STR'] + str(index) + '.pickle'
        save = {
            'test_data': data,
            'IMAGE_SIZE': dp['IMAGE_SIZE'], 
            'NUM_CHANNELS': dp['NUM_CHANNELS']
        }
        
        save_pickle_file(pickle_file, save, dp)

def save_pickle_file(pickle_file, save_dict, dp):
    try:
        if not os.path.exists(dp['PICKLE_PATH']):
                os.makedirs(dp['PICKLE_PATH'])
        f = open(dp['PICKLE_PATH'] + pickle_file, 'wb')
        pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

    print "Datasets saved to file", dp['PICKLE_PATH'] + pickle_file

if __name__ == '__main__':
    
    # Argument parse structure
    parser = argparse.ArgumentParser(description="Run a convolutional autoencoder on image data. Outputs activations.csv (the channel activations at each timestep), \
            norm_activation.csv (the normalized activations), perplexity.csv (the image reconstruction error at each timestep) and writes the latent channel activations  \
            to the directory specified by intermed_path and the reconstructed image to the path specified by recon_path.")
    parser.add_argument("-i", "--image_size", type = int, default = 400, help = "Size of the downsampled images, in pixels by pixels.")
    parser.add_argument("-c", "--image_channels", type = int, default = 3, help = "Number of color channels in image.")
    parser.add_argument("-t", "--image_depth", type = int, default = 255, help = "Value range of a pixel (0 - image_depth). Std = 255.")
    parser.add_argument("-f", "--file_chunk", type = int, default = 100, help = "Number of images saved in each pickle file.")
    parser.add_argument("-p", "--pickle_path", default = os.path.join(root_dir, 'pickle/'), help = "Path to store the pickle files.")
    parser.add_argument("-x", "--pickle_prefix", default = 'missionI.400.all.', help = "Prefix of the pickle files containing the data.")
    parser.add_argument("-d", "--data_path", default = os.path.join(root_dir,'images'), help = "Directory where the image files can be found.")
    args = parser.parse_args()
    
    print "Making dataset and saving it to:", args.pickle_path 
    print "To change this and other settings, edit the flags."

    data_params = {}
    data_params['IMAGE_SIZE'] = args.image_size
    data_params['NUM_CHANNELS'] = args.image_channels
    data_params['PIXEL_DEPTH'] = args.image_depth
    data_params['FILE_CHUNK'] = args.file_chunk
    data_params['PICKLE_PATH'] = args.pickle_path
    data_params['PREFIX_STR'] = args.pickle_prefix
    data_params['DATA_PATH'] = args.data_path


    make_basic_datasets(data_params)
  	
