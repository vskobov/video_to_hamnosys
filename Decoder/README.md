
### Create animation frames using JASigning Software

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

### Extract body keypoints from the frames using OpenPose

* `DATA_PATH` - path to directory with generated signs

Please use the installation guide provided with [OpenPose Software](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
Install OpenPose on the experimental machine, preferably use the one with GPU
place `Decoder/run_o_pose.sh` script under `openpose` directory

```
cd /openpose
bash ./run_o_pose.sh DATA_PATH/

```
## Data Preparation
### Data preparation for the training and annotation

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

### Make Annotations

* `MODEL_PATH` - path to trained model
* `REAL_DATA_PATH` - path to `HDF5` file with real data

You need to redo steps 1-4 passing the path to the data that is need to be annotated and after run:

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