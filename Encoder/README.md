Please install the requirements:
```
pip3 install -r requirements.txt
```
## Tree Creation

* The tree building rules can be viewed in rules.txt file
* The rule can be modified only in the corresponding .nw file

usage: SiGML generation Tree builder `-h -v`

This program builds generation Tree from the Rules in `Tree_Rules` folder

optional arguments:

* `-h, --help`     show this help message and exit
* `-v, --version`  show program's version number and exit

The tree is already created and stored in `Created_Trees` directory, but you are free to rebuild it again. 

# To rebuild the generation Tree use:

```
python3 build_a_tree.py
```
!!this can take up to 1 hour of computing time 


## Sign Generation

To generate signs after the tree is buld use  sigml_sign_generator.py

usage: SiGML Sign Generator `[-h] [-v] [-p] [-s] SIGNS_NUM SIGNS_BUNCH [OUTPUT_DEST]`
This program computes generates valid HamNoSys 4.0 Signs and outputs them in SiGML files

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



## Quick test


For quick testing we would recommend to generate 10 signs with 1 pro file:
```
python3 sigml_sign_generator.py 10 1 quick_10_signs -rs
                                                    -wrs
                                                    -mix
                                                          -p -s
```
To validate them, you can use online JASigning system:

please note, that online system has limitations on input size 
please also make sure that `<sigml></sigml>` and other xml brackets are closed 
if you copied a several signs from the sets files.

[Test SiGML online](#http://vhg.cmp.uea.ac.uk/tech/jas/vhg2018/WebGLAv.html)


## Expamples of data set creation


It is presented 10,000 sigs pro data set, due to submission size limitation

To recreate the Data Random Selection Data set use:
```
python3 sigml_sign_generator.py 10000 20 10000_RS -rs -s -p
```
this can take up to 8 hours of computing time

To recreate the Data Weighted Random Selection Data set use:
```
python3 sigml_sign_generator.py 10000 20 10000_WRS -wrs -s -p
```
this can take up to 16 hours of computing time

To recreate the Mixed Selection Data set use:
```
python3 sigml_sign_generator.py 10000 20 10000_MIX -mix -s -p
```
this can take up to 14 hours of computing time 
