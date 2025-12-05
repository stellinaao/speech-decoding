## Pytorch implementation of [Neural Sequence Decoder](https://github.com/fwillett/speechBCI/tree/main/NeuralDecoder)

## Requirements
- python >= 3.9

## Installation

pip install -e .

## How to run

1. Convert the speech BCI dataset using [formatCompetitionData.ipynb](./notebooks/formatCompetitionData.ipynb)
2. In the notebook, run one of the three dimensionality reduction algorithms under the dimensionality reduction section
3. Train model: `python ./scripts/train_model.py`
4. Toggle between LSTM and GRU with the RNN parameter

