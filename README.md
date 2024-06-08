# DDPM Implementation

My DDPM (Denoising Diffusion Probabilistic Model) implementation using PyTorch.

This project is developing... Just wait for that. 🤫

---

Original research paper: https://arxiv.org/pdf/2006.11239 (_Denoising Diffusion Probabilistic Models_)

Dataset from kaggle: https://www.kaggle.com/datasets/splcher/animefacedataset

---

## Preparations

### Download and Setup Dataset

- Run `data/download.sh`, this will download dataset from kaggle and unzip it.
- Run `python data/index.py`, this will generate two index JSON file for training.

Then you will see `train-index.json` and `val-index.json` appear in the `data` directory.
They're required for training.

## Training

### Start Training from Scratch

~~~
python train.py
~~~

Checkpoints will be saved in `checkpoints` directory.

**Arguments of train.py:**
- `-b`, ( _int_ ) Batch size.
- `-l`, (_float_) Learning rate.
- `-e`, ( _int_ ) Epochs to train.
- `-w`, ( _int_ ) How many dataloader workers.
- `-f`, ( _str_ ) Path to checkpoint.
- `-s`, ( _int_ ) How many iterations to save.
- `-t`, ( _int_ ) How many time steps of Markov chain.

### Continue Training

~~~
python train.py -f ./checkpoints/your-checkpoint.pth
~~~

## Inference

### Start Inference

~~~
python infer.py -f ./checkpoints/your-checkpoint.pth
~~~

Generated images will be saved in `out` directory by default.

**Arguments of infer.py:**
- `-f`, ( _str_ ) Path to checkpoint.
- `-o`, ( _str_ ) Output directory.
- `-b`, ( _int_ ) Batch size.
- `-n`, ( _int_ ) How many images to generate.
- `-x`, ( _int_ ) Width of generated image.
- `-y`, ( _int_ ) Height of generated image.
- `-t`, ( _int_ ) How many time steps of Markov chain.
