# particleSeg
`particleSeg` is a deep learning-based segmentation tool for cryo-EM particles.
![work_flow](images/workflow.png)

## Installation
1. Make sure you have [conda](https://docs.conda.io/en/latest/miniconda.html), [CUDA](https://developer.nvidia.com/cuda-toolkit)(version >= 11.0), and [cuDNN](https://developer.nvidia.com/cudnn) installed
2. Create conda environment
```
conda env create -f env_particleSeg.yml
```
3. Install dependencies from pip
```
# activate the environment just installed
conda activate particleSeg

# install dependencies using pip
pip install -r requirement.txt

# install tensorflow_example from github
pip install -q git+https://github.com/tensorflow/examples.git
```
## Usage
```
usage: segParticles.py [-h] [--train TRAIN] [--infer INFER] [--raw_data RAW_DATA]
ParticleSeg: Segment your particles
optional arguments:
  -h, --help           show this help message and exit
  --train TRAIN        Train models with current dataset.
  --infer INFER        Take a metadata(starfile) and return cleaned metadata.
  --raw_data RAW_DATA  Take raw data directory(where *.mrcs stored) if metadata for inference is provided.
```
Train all pretrained models provided 25600 particles to test performance of each model
```
python segParticles.py --train all
```
Train a specified model with all particles
```
python segParticles.py --train DenseNet169
```
Inference your particles
```
python segParticles.py --infer /path/to/your/particles.star --raw_data /path/to/your/raw/
```
## Pretrained custom models
```
base_models = [
  'custom',
  'DenseNet121',
  'DenseNet169',
  'DenseNet201',
  'EfficientNetB0',
  'ResNet101'
]
```
## Getting Help
Please contact nzhou26@outlook.com 
