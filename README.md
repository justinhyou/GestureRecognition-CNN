# Image Classification: Fine-Tune CNN Model via Transfer Learning
Transfer Learning CNN for multi-class classification of surgical gestures based on JIGSAW dataset, as well as dual classification for handwritten vs typed and cats vs dogs.

## Requirements:
This project was built using Ubuntu 16.04, Ananconda, Keras, and Tensorflow. The code has been tested with both CPU and GPU. Keras 1.2.2 was used. For MacOS, Tensorflow no longer support GPU. 

## Run:
1. Download [__Anaconda__](https://www.continuum.io/downloads)
2. Download JIGSAW dataset. 
3. Run script to produce data folder containing "train" and "validation" subfolders.*
4. Download the current configuration for a model. This model is editable in a JSON format. 

*For own dataset, make sure your files are in ../data/ with two folders (/validation/ and /test/) where each contains folders for each class to be clasified (i.e. /data/validation/some_class/some_image.jpg).


### Data Preparation 
Set the environment variable DATA_HOME to the directory where your data folder lives

(Optional) Create downsampled, square image pickle files using make_dataset.py
1. Place all of your images in a folder within your DATA_HOME directory. 
2. Open ```make_dataset.py``` and set USER_KEY in the user configurables section to a function object that takes a filename as input and returns an interger that allows the files to be sorted chronologically.
3. Run make_dataset.py, and use the flags to specify options. The ```--help``` flag displays parameter options. Set the following flags:
    - ```--image_size```: the desried downsampled image size, in pixels x pixels
    - ```--image_channels```: dimension of the origigonal image. 3 for RGB images, 1 for greyscale
    - ```--pickle_prefix```: the range of pixels values, 0 - PIXEL_DEPTH. Standard: 255
    - ```--file_chunk```: number of image files per pickle file. If you choose a large IMAGE_SIZE, you may have to choose a smaller FILE_CHUNK
    - ```--pickle_path```: where you want to store the resulting pickle files
    - ```--data_path```: where the image files can be found
4. Run ``python make_dataset.py`` to create a dataset
5. Final pickle files will be saved to the dir specified in PICKLE_PATH

## Code:
The code contains two training model, and one classification output.
1. fine_tune.py: used to train CNN to classify between dogs vs. cats.
2. fine_tune_text_images: used to train CNN to classify between handwritten vs. typed.
3. classify.py: used to classify new samples for either training model

```sh
python code/fine_tune.py <data_dir/> <model_dir/>
```
* Make sure to include the `/` at the end of every directory for the example to work.

The training script will save the json model `model.json`, and `model_weights.h5` file in the specified <model_dir/>
The classify script will save a `predictions.csv` file in the specified <results_dir/>

## Model:
This directory contains the best model already trained on JIGSAW data.

## Results:
This directory contains the test results for both classification tasks.

__Best Value__:
* Dog vs. Cats: 99.38% validation accuracy, .02 validation log loss
* Handwritten vs Typed: 100% validation accuracy
* JIGSAW dataset (video only): 92% validation accuracy (80/20 random split of train/validation)

## Data:
This empty directory is created to store the dataset if desired. Download the dataset from the requirements section and place it inside this folder.
_Example:_ `data/dogs_cats/..`

## Acknowledgement:
A significant portion of this script came from keras blog example: [_"Building powerful image classification models using very little data"_](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
My main contribution is to make it work with all keras pre-train applications models, and add a higher level of abstraction.
