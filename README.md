# DLCV Final Project ( Multiple Concept Personalization )

## Create Conda Environment
```shell script=
conda create -n dlcv-final python=3.9
conda activate dlcv-final
pip install -r requirements.txt
```

## Generate Images
```shell script=
bash ./scripts/download.sh # download ckpt and dataset
bash ./scripts/main.sh # generate images
```

## Evaluation
```shell script=
bash ./scripts/evaluate.sh
```