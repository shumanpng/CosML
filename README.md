# CosML: Combining Domain-Specific Meta-Learners

## Environment Setup
Requirements:
```
Python 3.6
PyTorch 1.2
```

```
virtualenv --python=/usr/bin/python3.6 dgml_venv_py36
source dgml_venv_py36/bin/activate
pip install -r requirements.txt
```

## Datasets
To download and preprocess the data for the Mini-Imagenet, Cars, CUB, Places, and Plantae datasets, please follow the instructions from [Tseng el al. (2020)](https://arxiv.org/abs/2001.08735)'s GitHub repo [here](https://github.com/hytseng0509/CrossDomainFewShot#datasets).


## Pre-trained 5w5s CosML Models

Please download and unzip all the models in the `cosml_project/cosml/output` directory.

### Download Models
Please use the links below to download trained models for cross-domain few-shot classification (5w5s) on the specified unseen dataset:
* [unseen Cars](https://drive.google.com/file/d/11Kk6WgXxp9rSP7IisIiIIhPOtluVgiFU/view?usp=sharing)
* [unseen CUB](https://drive.google.com/file/d/1lqzj5sVyurL2hVgw2P94K6Jt-qbTJ_ef/view?usp=sharing)
* [unseen Places](https://drive.google.com/file/d/188oY2S9rcJXRjwqqLl4jJPIFzCr9MQ6o/view?usp=sharing)
* [unseen Plantae](https://drive.google.com/file/d/1LyTJu_ITPlRniwf3zr82uZXmSVEPpwhi/view?usp=sharing)


### Testing Pre-trained Models:

First, make sure that you are inside the `cosml` directory:
```
cd cosml/

```

| Unseen Domain   | Accuracy         | Command           |
|-----------------|:----------------:|:------------------|
| Cars            | 60.17 +/- 0.63%  | `bash test_pretrained_cosml_cars.sh`
| CUB             | 66.15 +/- 0.63%  | `bash test_pretrained_cosml_cub.sh`
| Places          | 88.08 +/- 0.46%  | `bash test_pretrained_cosml_places.sh`
| Plantae         | 42.96 +/- 0.57%  | `bash test_pretrained_cosml_plantae.sh`





## Training CosML Models
First, make sure that you are inside the `cosml` directory:
```
cd cosml/

```

### Pre-training
The command below non-episodically trains a model on using the Mini-Imagenet dataset. Note that `[PATH TO DATASETS]` should be the path to the parent directly where all 5 datasets are stored (it should end in `CrossDomainFewShot-master/filelists/`)
```
python pretrain.py --testset cub,cars,places,plantae --data_dir [PATH TO DATASETS]

# e.g.
python pretrain.py --testset cub,cars,places,plantae --data_dir ../../CrossDomainFewShot-master/filelists/

```

### Meta-training
Inside `train_cosml.sh`, you may wish to update `TESTSET` and `DATADIR` depending on which model you want to train and where your downloaded data is stored.

To train a model to perform cross-domain few-shot classification on the unseen dataset `Cars`, `TESTSET` should be set to `cars`. The resulting model is named `cosml_miniImagenet_cub_places_plantae_5w5s_conv4_conv2+linear_euclidean`, which is located in `output/checkpoints/`. This model name will be the name you specific in `test_cosml.sh` if you wish to test this model.

After modifying `TESTSET` and `DATADIR`, you can train the model using the following command:
```
bash train_cosml.sh

```

### Meta-testing
```
bash test_cosml.sh

```


## Note
This code is built upon the implementation from [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot) and [CrossDomainFewShot](https://github.com/hytseng0509/CrossDomainFewShot). We would like to thank the authors of CloserLookFewShot and CrossDomainFewShot for kindly making their implementations publicly available.
