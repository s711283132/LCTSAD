# LCTSAD
PVLDB

# Installation

Environment

Using Python 3.7.16

# Train & Test

Run the model on a dataset with:
```bash
python3 main.py --dataset <dataset>
```
Available datasets :
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


