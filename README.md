# LCTSAD
PVLDB

# Installation

## Environment
- Python 3.7.16 recommended

## Install Dependencies
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

# Train & Test

Run the model on a dataset with:
```bash
python3 main.py --dataset <dataset>
```
## Available datasets
- MBA
- MSDS
MBA MSDS MSL NAB SMAP SMD SWaT UCR WADI

To run the model on the MBA dataset:
```bash
python3 main.py --dataset MBA
```
# Benchmmark
To evaluate the model latency on a specific dataset, run:
```bash
python3 benchmark.py --dataset <dataset>
```


