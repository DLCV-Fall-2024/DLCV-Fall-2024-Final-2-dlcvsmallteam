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

## Note

- This takes about 4 hours on RTX 4090. (Because we generate 100 images for selection)
- This is for codalab

## Peer Review Images

For prompt 0 to prompt 2, please check the flux_gen directory.

```shell script=
cd ./flux_gen
```

For prmopt 3, please see the output codalab_output (after generating images for codalab_output)
