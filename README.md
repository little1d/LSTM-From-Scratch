## LSTM-From-Scratch
Learn LSTM Neural Network From Scratch!

This repository provides a step-by-step guide to building an LSTM (Long Short-Term Memory) neural network from scratch. We demonstrate the use of our model on a Google stock price prediction task and visualize the results using the SwanLab tool.

## How To Install Environment
If you don't have GPU support, you can install the environment by running the following command:

```python
conda create -n lstm python==3.10

conda activate lstm

pip install uv && uv pip install -r requirements.txt

```

## Project Overview
This project aims to provide a comprehensive understanding of how LSTM networks work by building one from scratch. The main components of the project include:
```
├── docs
│   └── lstm.md
├── notebook
│   ├── GOOG.csv
│   ├── lstm.ipynb
│   ├── model.py
│   ├── simple_implement.py
│   └── main.py
└── requirements.txt
```
* Model Definition: The `lstm.ipynb/model.py` file contains the implementation of the LSTM model from scratch.

* Simple Implementation: The `simple_implement.py` file demonstrates how to use LSTM model cope with stock prediction problem using PyTorch's nn.Module library, along with evaluating and visualizations

* Data Handling, Model Traning: The `main.py` includes functions for loading and preprocessing the Google stock price data, a step-by-step guide to training the lstm model using **swanlab**


## Acknowledgements
**This repo is inspired by the great work I've seen in swanlab tutorial.**
* open source swanlab repo: [https://github.com/SwanHubX/SwanLab](https://github.com/SwanHubX/SwanLab)

* uv: [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)

* Dive Into Deep Learning: [https://zh.d2l.ai/chapter_installation/index.html](https://zh.d2l.ai/chapter_installation/index.html)

## My Traning log in SwanLab
[https://swanlab.cn/@Harrison/Google-Stock-Prediction/overview](https://swanlab.cn/@Harrison/Google-Stock-Prediction/overview)