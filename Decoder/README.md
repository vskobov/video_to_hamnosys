# Master Research

Title: Automatic HamNoSys annotation of video-based Sign Language Corpus using Machine Learning

Author: Victor Skobov (v.skobov@fuji.waseda.jp)

Abstract: The Hamburg Notation System (HamNoSys) was developed for movement annotation of any sign language (SL) and can be used to produce signing animations for a virtual avatar with the JASigning platform. This provides the potential to use HamNoSys, i.e., strings of characters, as a representation of an SL corpus instead of video material. Processing strings of characters instead of images can significantly contribute to sign language research. However, the complexity of HamNoSys makes it difficult to annotate without a lot of time and effort. Therefore annotation has to be automatized. This work proposes a conceptually new approach to this problem. It includes a new tree representation of the HamNoSys grammar that serves as a basis for the generation of grammatical training data and classification of complex movements using machine learning. Our proposed system is capable of producing grammatically correct annotations of any sign in any sign language and can potentially be used on already existing SL corpora. It is able to correctly transcribe 54\,\% of random signing handshapes from a validation set. It is retrainable for specific settings such as camera angles, speed, and gestures. Our approach is conceptually different from other SL recognition solutions and offers a developed methodology for future research.

Link to Thesis:

Publications:

# Table of Contents
* Documentation
* Requirements
* Quickstart
* Acknowledgements
* Citation

## Documentation

## Requirements

Install required packages by using `pip` with `requirements.txt`:

```
pip install -r requirements.txt
```

Note:

## Quick start

### Generate the training Data

```
```

### Data preparation for the training and annotation

```
```


### Train model

```
python nn_train.py DATA_LOCATION
```

### Make Annotations

```
```
