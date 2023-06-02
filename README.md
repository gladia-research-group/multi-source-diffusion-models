# Multi Source Diffusion Models
Official repository for [Multi-Source Diffusion Models for Simultaneous Music Generation and Separation](https://arxiv.org/abs/2302.02257).

Demo website available [here](https://gladia-research-group.github.io/multi-source-diffusion-models/).


# Installation
The environment for running our code can be installed using conda:
```bash
# Install environment
conda env create -f env.yaml

# Activate the environment
conda activate msdm
```

## Download dataset
This step is optional and required only if you are interested in training a model from scratch or reproducing the results from the paper. 

Detailed instructions to download our dataset can be found [here](data/README.md).

## Download pretrained checkpoints
It is possible to download the pretrained models we used in our experiments. These are useful to run the generation, inpainting and separation scripts available in this repository. 

Detailed instruction to download our pretrained models can be found [here](ckpts/README.md). 

# Training

> ⚠️ **NOTE:**  
> Before executing the training script, you have to rename the file `.env.tmp` to `.env` and change the **WANDB_\*** environment variables to match your 
personal or institutional account.

You can run the model by invoking the `train.py` script:
```bash
# Activate environment
conda activate msdm

# Run training script
PYTHONPATH=. python train.py exp=train_msdm
```
The results will be logged into your **Weight & Biases** account, using the information provided before.

# Generation and Inpainting

## Run Notebook
Take a look at the notebook `notebook/MSDM-generation.ipynb` for generation and inpainting examples.

> ⚠️ **NOTE:**  
> Before running the notevook, download the `glorious-star-335` pretrained checkpoint.

The notebook can be executed using the command
```bash
jupyter notebook notebooks/MSDM-generation.ipynb
```
The example found in our [demo website](ttps://gladia-research-group.github.io/multi-source-diffusion-models/)
 were produced using a similar notebook.

# Separation evaluation
> ⚠️ **NOTE:**   
> You need to download the test set and the pretrained checkpoints in order to run the evaluation scripts.

Running the script
```bash
# Run evaluation script for MSDM model
PYTHONPATH=. python evaluate.py exp=eval_msdm_dirac
```
will separate the test set and evaluate its quality using the SI-SNRi metric. After executing the script, the separation results can be found inside of the `output/separations/<exp-name>` directory. The metrics are reported inside of the `metrics.csv` file, in the separation folder.

It is possible to use different configuration for the separation script, by changing the `exp=<config>` input argument.
You can choose between the following configurations:
```bash
exp=eval_msdm_dirac # < second best performance
exp=eval_msdm_gaussian
exp=eval_weakly_msdm_dirac # < best performance (4x slower)
exp=eval_weakly_msdm_gaussian
```