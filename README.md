# Automatic HamNoSys annotation of video-based Sign Language Corpus using Machine Learning

**Author**: Victor Skobov (v.skobov@fuji.waseda.jp)

**Abstract**: 
*The Hamburg Notation System (HamNoSys) was developed for movement annotation of any sign language (SL) and can be used to produce signing animations for a virtual avatar with the JASigning platform. This provides the potential to use HamNoSys, i.e., strings of characters, as a representation of an SL corpus instead of video material. Processing strings of characters instead of images can significantly contribute to sign language research. However, the complexity of HamNoSys makes it difficult to annotate without a lot of time and effort. Therefore annotation has to be automatized. This work proposes a conceptually new approach to this problem. It includes a new tree representation of the HamNoSys grammar that serves as a basis for the generation of grammatical training data and classification of complex movements using machine learning. Our proposed system is capable of producing grammatically correct annotations of any sign in any sign language and can potentially be used on already existing SL corpora. It is able to correctly transcribe 54% of random signing handshapes from a validation set. It is retrainable for specific settings such as camera angles, speed, and gestures. Our approach is conceptually different from other SL recognition solutions and offers a developed methodology for future research.*

Link to Thesis:()

Publications:()

# Table of Contents
* [Requirements](#requirements)
* [Quickstart](#quickstart)
* [Data Generation](#train-model)
* [Data Preparation](#data-preparation)
* [Train Models](#train-model)
* [Produce annotations](#produce-annotations)

## Requirements
Research relies heavily on external libraries and tools:

* [JASigning Software](http://vh.cmp.uea.ac.uk/index.php/JASigning)
* [OpenPose Software](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

Python packages: Install required packages by using `pip` with `requirements.txt`:

```
pip install -r requirements.txt
```
## Quickstart

## Data Generation
### 1 Generate the training Data
To generate signs after the tree is buld use  `sigml_sign_generator.py`

usage: SiGML Sign Generator `[-h] [-v] [-p] [-s]` `SIGNS_NUM` `SIGNS_BUNCH` `[OUTPUT_DEST]`

This program computes generates valid HamNoSys 4.0 Signs and outputs them in
SiGML files

positional arguments:

* `SIGNS_NUM`      Number of Signs to generate
* `SIGNS_BUNCH`    Amount of signs stored in one SiGML file
* `OUTPUT_DEST`    Output file directory (ex : ./folder/)


optional arguments:
* `-h`, `--help`     show this help message and exit
* `-wrs`           Use this option to apply Weighted Random Selection of optional nodes during the generation
* `-rs`            Use this option to apply Random Selection of optional nodes during the generation, it is set as default method
* `-mix`           Use this option to apply mix of Weighted Random Selection and Random Selection of optional nodes during the generation
* `-v`, `--version`  show program's version number and exit
* `-p`, `--plot`     Use this option to output a descriptive statistics plot
* `-s`, `--symbols`  Use this option to print symbols usage and mentions

        
```
python3 sigml_sign_generator.py 5 1 DATA_PATH -wrs

```
To validate them, you can use online JASigning system: [Online SiGML animation](#http://vhg.cmp.uea.ac.uk/tech/jas/vhg2018/WebGLAv.html)

* please note, that online system has limitations on input size 
* please also make sure that <sigml></sigml> and other xml brackets are closed 
* if you copied a several signs from the sets files.


### 2 Create animation frames using JASigning Software

* `DATA_PATH` - path to directory with generated signs

Automatic sigml data generation is possible only under **macOS**, due to JASigning Software limitation
After dowloding `jas.zip`, using [Netbeans](#https://netbeans.org/) you can open and build the JASApp
Please follow the instructions on the [JASigning Software](http://vh.cmp.uea.ac.uk/index.php/JASigning) paige

Place `Decoder/run.sh` script under `/jas/loc2018/JASApp` directory

First run the reciever in the **separate** shell:
```
python3 s_reciever.py

```

Now in the **other** shell:

```
mkdir DATA_PATH/All_Frames
mkdir DATA_PATH/All_Keys

cd /jas/loc2018/JASApp
bash ./run.sh DATA_PATH

```

### 3 Extract body keypoints from the frames using OpenPose

* `DATA_PATH` - path to directory with generated signs

Please use the installation guide provided with [OpenPose Software](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
Install OpenPose on the experimental machine, preferably use the one with GPU
place `Decoder/run_o_pose.sh` script under `openpose` directory

```
cd /openpose
bash ./run_o_pose.sh DATA_PATH

```
## Data Preparation
### 4 Data preparation for the training and annotation

* `DATA_PATH` - path to directory with generated signs

first let's move extracted keys back to the sign directories
```
python3 move_extracted_keys.py DATA_PATH

```

now let's process the keys and create `numpy` arrays as input for each sign
and store all arrays as `HDF5` training and test files

```
python3 framekeys_to_ndarray.py DATA_PATH
python3 store_h5_data.py DATA_PATH

```

## Train Models

* `DATA_PATH` - path to directory with training and test `HDF5` files

after creating `HDF5` training and test files you can move them to separate directory `Sign_Data` or else
### 5 Train the models and rebuild the tree
this will create `models` and `models/nodes` directory in the `DATA_PATH` where the trained tree will be stored

Trained model will be placed under: `models/3_nn_multi_train.joblib`
```
python3 nn_train.py DATA_PATH
```
## Produce annotations

### 6 Make Annotations

* `MODEL_PATH` - path to trained model
* `REAL_DATA_PATH` - path to `HDF5` file with real data

You need to redo steps 1-4 passing the path to the data that is need to be annotated and after run:

```
python3 annotate.py REAL_DATA_PATH MODEL_PATH

```
