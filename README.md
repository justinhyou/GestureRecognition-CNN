# Image Classification: Fine-Tune CNN Model via Transfer Learning
Transfer Learning CNN for multi-class classification of surgical gestures based on JIGSAW dataset. 

## Requirements:
This project was built using Ubuntu 16.04, Ananconda, Keras, and Tensorflow. The code has been tested with both CPU and GPU. 

## Run:
1. Download [__Anaconda__](https://www.continuum.io/downloads)
2. Provide data in a "Data" directory with "train" and "validation" subfolders. Each subfolder should contain a folder for each image class with a sizeable number of images per.* 
3. Download the current configuration for a model. This model is editable in a JSON format. 

*For own dataset, make sure your files are in ../data/ with two folders (/validation/ and /test/) where each contains folders for each class to be clasified (i.e. /data/validation/some_class/some_image.jpg).

## Code:
The code contains two training model, and one classification output.
1. fine_tune.py: used to train CNN to classify between dogs vs. cats.
2. fine_tune_text_images: used to train CNN to classify between handwritten vs. typed.
3. classify.py: used to classify new samples for either training model

## Training: Dogs vs. Cats Example
Both the dogs_cats and text_images classifier have already been trained and their best models will be saved on the model directory after downloading them [_models download_](https://my.pcloud.com/publink/show?code=VZvzMlZyPYO1hSn92LXdzmGNr9y1j7qKDzX). If you do not wish to re-train the models feel free to skip this step and go straight into classifying new samples.

```sh
image-classifier$ source activate image-classifier-cpu
image-classifier$ python code/fine_tune.py <data_dir/> <model_dir/>
```
_Example:_ `python code/fine_tune.py data/dogs_cats/ model/dogs_cats/`

* Make sure to include the `/` at the end of every directory for the example to work.
* Replace __image-classifier-cpu__ with __image-classifier-gpu__ if your using GPU

The training script will save the json model `model.json`, and `model_weights.h5` file in the specified <model_dir/>

## Classify: Dogs vs. Cats Example
For text_images dataset parameters are `nb_test_samples`=10, `img_width`=600, `img_height`=150 must be change according to your model and needs. Values provided are just default ones.

For dogs vs cats dataset just leave default values.

```sh
image-classifier$ source activate image-classifier-cpu
image-classifier$ python code/classify.py <model_dir/> <test_dir/> <results_dir/>
```

_Example:_ `python code/classify.py model/dogs_cats/ data/dogs_cats/test/ results/dogs_cats/`

* Make sure to include the `/` at the end of every directory for the example to work.
* Replace __image-classifier-cpu__ with __image-classifier-gpu__ if your using GPU
* `<test_dir/>` should contain a subfolder inside and NOT the data directly. Example: `test_dir/test`

The classify script will save a `predictions.csv` file in the specified <results_dir/>

## Model:
This directory contains the best models already train for both classification tasks.

## Results:
This directory contains the test results for both classification tasks.

__Best Value__:
* Dog vs. Cats: 99.38% validation accuracy, .02 validation log loss
* Handwritten vs Typed: 100% validation accuracy


## Data:
This empty directory is created to store the dataset if desired. Download the dataset from the requirements section and place it inside this folder.
_Example:_ `data/dogs_cats/..`

## Acknowledgement:
A significant portion of this script came from keras blog example: [_"Building powerful image classification models using very little data"_](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
My main contribution is to make it work with all keras pre-train applications models, and add a higher level of abstraction.
