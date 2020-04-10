# Pix2Pix for Fruits

This is a mini project where is try to try pix2pix myself, but on fruits. Dataset used is from [COCO-dataset](http://cocodataset.org/#home)

You can find the original paper [here](https://phillipi.github.io/pix2pix/).

### Setting up environment

Take note that i am using ubuntu

1. Install anaconda
1. On your terminal, run `conda create env -f environment.yml`
1. On your terminal. run `conda activate pix2pix_env`

### Generating Data

1. Make sure that you have activated the conda environment
1. Download COCO training and validation dataset and annotations
1. Edit the following variables inside **process_data.py**
```
training_data_dir
validation_data_dir
training_annotation_file
validation_annotation_file
```
1. On your terminal, run `python proces_data.py`
1. The images will now be saved inside the `train` and `val` folders
