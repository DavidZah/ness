# Neural Network Training Framework

This repository contains a framework for training neural networks using custom hyperparameter combinations and configurations. It supports parallel execution, detailed logging, progress tracking, and easy customization of datasets and network architectures.

## Features

- **Automated Hyperparameter Grid Search:** Explore combinations of optimizers, activation functions, layer sizes, neuron counts, and learning rates.
- **Parallel Execution:** Configurable parallel execution to maximize resource utilization.
- **WandB Integration:** Log training and validation metrics, visualize results, and monitor training in real-time.
- **Custom DataLoader:** Flexible data loading, shuffling, and splitting for training, validation, and testing.
- **Early Stopping:** Prevent overfitting by stopping training when validation loss stops improving.
- **Visualizations:** Generate classification plots for training and validation datasets.

## Project Structure

```plaintext
.
├── data/
│   ├── loader_data/                # Preprocessed datasets for training
│   └── tren_data/                  # Raw datasets
├── logs/                           # Logs and saved weights
├── main.py                         # Training script
├── dataloader.py                   # DataLoader for handling datasets
├── run.sh                          # Bash script to automate training runs
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- `pip` for managing dependencies

Install dependencies using:

```bash
pip install -r requirements.txt
```

### Setup

1. Place raw datasets in the `data/tren_data/` folder.
2. Preprocess the datasets:

   ```bash
   python preprocess.py
   ```

   This script splits the data into training, validation, and test sets and saves them in `data/loader_data/`.

3. Adjust the parameters in `run.sh` as needed (e.g., dataset path, number of processes).

## Usage

### Run Training Script

To run a single training session:

```bash
python main.py \
    --num_layers 4 \
    --layer_width 64 \
    --optimizer adam \
    --activation relu \
    --epochs 100 \
    --learning_rate 0.001 \
    --dataset tren_data1 \
    --name Experiment-1
```

### Automated Hyperparameter Tuning

To run multiple experiments with a grid search over hyperparameters:

```bash
bash run.sh
```

Modify `run.sh` to set the hyperparameters, dataset, and logging directory.

## Parameters

### Training Script (`main.py`)

| Parameter          | Description                                | Default      |
|--------------------|--------------------------------------------|--------------|
| `--num_layers`     | Number of layers in the neural network     | `4`          |
| `--layer_width`    | Number of neurons per layer                | `64`         |
| `--optimizer`      | Optimizer (`sgd` or `adam`)                | `adam`       |
| `--activation`     | Activation function (`relu` or `sigmoid`) | `relu`       |
| `--epochs`         | Number of training epochs                  | `1000`       |
| `--learning_rate`  | Learning rate                              | `0.01`       |
| `--dataset`        | Dataset to use                            | `tren_data1` |
| `--name`           | Name for the training session             | `Experiment-1` |

## Visualization

Training and validation metrics are logged to [WandB](https://wandb.ai). You can visualize performance, track experiments, and compare results easily.

Example:

```bash
wandb login
```

## Contribution

Feel free to contribute by:

- Improving documentation
- Adding support for additional optimizers or activation functions
- Enhancing the visualization capabilities

## License

This project is licensed under the MIT License.

---
