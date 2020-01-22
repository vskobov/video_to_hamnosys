# Automatic HamNoSys annotation of video-based Sign Language Corpus using Machine Learning

**Author**: Victor Skobov (v.skobov@fuji.waseda.jp)

**Abstract**: 
*The Hamburg Notation System (HamNoSys) was developed for movement annotation of any sign language (SL) and can be used to produce signing animations for a virtual avatar with the JASigning platform. This provides the potential to use HamNoSys, i.e., strings of characters, as a representation of an SL corpus instead of video material. Processing strings of characters instead of images can significantly contribute to sign language research. However, the complexity of HamNoSys makes it difficult to annotate without a lot of time and effort. Therefore annotation has to be automatized. This work proposes a conceptually new approach to this problem. It includes a new tree representation of the HamNoSys grammar that serves as a basis for the generation of grammatical training data and classification of complex movements using machine learning. Our proposed system is capable of producing grammatically correct annotations of any sign in any sign language and can potentially be used on already existing SL corpora. It is able to correctly transcribe 54% of random signing handshapes from a validation set. It is retrainable for specific settings such as camera angles, speed, and gestures. Our approach is conceptually different from other SL recognition solutions and offers a developed methodology for future research.*

Link to [Thesis](http://133.9.48.111/files/Papers/Master_theses/Spring_2020/SKOBOV_Victor_Automatic_HamNoSys_annotation_of_video-based_Sign_Language_Corpus_using_Machine_Learning/SKOBOV_Victor_Automatic_HamNoSys_annotation_of_video-based_Sign_Language_Corpus_using_Machine_Learning.pdf)
Link to [Final Presentation](http://133.9.48.111/files/Papers/Master_theses/Spring_2020/SKOBOV_Victor_Automatic_HamNoSys_annotation_of_video-based_Sign_Language_Corpus_using_Machine_Learning/Skobov_SL_final.pdf)


# Table of Contents
* [Requirements](#requirements)
* [Quick Test](#quickstest)
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
python3 -m pip install -r requirements.txt --user
```
## Quickstest

`YOUR_HOME_DIRECTORY` is a full path to your home directory for ex. for me it was: `/home/Skobov_Victor/`

Login to Exp 12 machine and clone the project:

```
ssh your_name@exp12.local

git clone http://133.9.48.111:8082/SKOBOV_Victor/sl_annotation_research.git

python3 -m pip install -r sl_annotation_research/requirements.txt --user
```
you can install OpenPose and use your installation, then you need to place `Decoder/run_o_pose.sh` script under `openpose` directory

otherwise:
```
cd /home/Skobov_Victor/openpose

bash ./run_o_pose.sh YOUR_HOME_DIRECTORY/sl_annotation_research/sample_dev_set_hand_conf_wrs_5/
```
now the body keyoints are extracted
we will prepare them as trainig and testing data and train a simple tree with `EPOCHS=2` and  `BATCH_SIZE=1`
training and tree rebuilding will take aprx. 1 hour

```
cd YOUR_HOME_DIRECTORY/sl_annotation_research/Decoder

python3 move_extracted_keys.py YOUR_HOME_DIRECTORY/sl_annotation_research/sample_dev_set_hand_conf_wrs_5/
python3 framekeys_to_ndarray.py YOUR_HOME_DIRECTORY/sl_annotation_research/sample_dev_set_hand_conf_wrs_5/
python3 store_h5_data.py YOUR_HOME_DIRECTORY/sl_annotation_research/sample_dev_set_hand_conf_wrs_5/
python3 quick_nn_tree_train.py YOUR_HOME_DIRECTORY/sl_annotation_research/sample_dev_set_hand_conf_wrs_5/

```
after it is done it should return a result table with 0 results like this:
```
Accuracy across all 5728 Subclasses: 0 %
Results found 1658 Nodes without any training: 1658
...
...
\begin{tabular}{lrrrr}
\hline
 Tree level   &   SM &   Avg. N/SC &   Avg. SC &   Avg. Valid Accuracy \\
\hline
 All Levels   & 1658 &           0 &         3 &                   nan \\
\hline
\end{tabular}
```

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


First run the reciever in the **separate** shell:
```
python3 s_reciever.py

```

Now in the **other** shell:

```
mkdir DATA_PATH/All_Frames
mkdir DATA_PATH/All_Keys

cd /jas/loc2018/JASApp
bash ./Decoder/run.sh DATA_PATH/ /jas/loc2018/JASApp

```

### 3 Extract body keypoints from the frames using OpenPose

* `DATA_PATH` - path to directory with generated signs

Please use the installation guide provided with [OpenPose Software](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
Install OpenPose on the experimental machine, preferably use the one with GPU
place `Decoder/run_o_pose.sh` script under `openpose` directory

```
cd /openpose
bash ./run_o_pose.sh DATA_PATH/

```
## Data Preparation
### 4 Data preparation for the training and annotation

* `DATA_PATH` - path to directory with generated signs

first let's move extracted keys back to the sign directories
```
python3 move_extracted_keys.py DATA_PATH/

```

now let's process the keys and create `numpy` arrays as input for each sign
and store all arrays as `HDF5` training and test files

```
python3 framekeys_to_ndarray.py DATA_PATH/
python3 store_h5_data.py DATA_PATH/

```

## Train Models

* `DATA_PATH` - path to directory with training and test `HDF5` files

### 5 Train the models and rebuild the tree
this will create `models` and `models/nodes` directory in the `DATA_PATH` where the trained tree will be stored

Trained model will be placed under: `DATA_PATH/models/3_nn_multi_train.joblib`

Adjust `EPOCHS` and  `BATCH_SIZE` before training in `quick_nn_tree_train.py` file

```
python3 quick_nn_tree_train.py DATA_PATH
```
## Produce annotations

### 6 Make Annotations

* `MODEL_PATH` - path to trained model
* `REAL_DATA_PATH` - path to `HDF5` file with real data

You need to redo steps 1-4 (disable data shuffle in `store_h5_data.py` and `VALID_SET=0`) 
passing the path to the data that is need to be annotated and after run:


```
python3 annotate.py REAL_DATA_PATH MODEL_PATH

```

## Repeat my experiments

I highly recommend using Exp 14 
be aware that each of the experiments takes about 30+ hours on the Exp 14

```
python3 nn_tree_research_experiment_repeat.py /itigo/../LastFrame_Train_Data/
python3 nn_tree_research_experiment_repeat.py /itigo/../FiveFrames_Train_Data/
```
