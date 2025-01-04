import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import wandb
import argparse
import pickle
from dataloader import  CombinedDataloader
import matplotlib
matplotlib.use('Agg')
# Parse command-line arguments
parser = argparse.ArgumentParser(description="Neural Network Training Script")
parser.add_argument("--num_layers", type=int, default=4, help="Number of layers in the neural network")
parser.add_argument("--layer_width", type=int, default=8, help="Number of neurons in each layer")
parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam"], help="Optimizer to use (adam or sgd)")
parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid"], help="Activation function to use")
parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for training")
parser.add_argument("--early_stopping", type=int, default=10, help="Early stopping patience (in epochs)")
parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save weights and logs")
parser.add_argument("--dataset", type=str, default="tren_data1", help="Dataset to use for training")
parser.add_argument("--name", type=str, default="NN-Vanila-Training-run", help="Name of the WandB run")
args = parser.parse_args()

# Initialize WandB project
wandb.init(project="NN-Training-data2", name=args.name,config={
    "num_layers": args.num_layers,
    "layer_width": args.layer_width,
    "optimizer": args.optimizer,
    "activation": args.activation,
    "epochs": args.epochs,
    "learning_rate": args.learning_rate,
    "early_stopping": args.early_stopping
})


def relu_(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Activation function selection
if args.activation == "relu":
    activation_fcn = relu_
    activation_derivative = relu_derivative
elif args.activation == "sigmoid":
    activation_fcn = sigmoid
    activation_derivative = sigmoid_derivative

def one_hot_encode(y, num_classes):
    """Convert a vector of labels to one-hot encoding."""
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y.astype(int)] = 1
    return one_hot

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # For numerical stability
    return e_x / e_x.sum(axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]  # Avoid log(0) by adding 1e-9

def cross_entropy_derivative(y_true, y_pred):
    return y_pred - y_true

class DenseLayer:
    def __init__(self, size=None, weights=None, activation_fcn=None, weights_init="rand"):
        if size is None and weights is None:
            raise Exception("Specify either shape or weights.")
        if weights is None:
            if weights_init == "zeros":
                self.weights = np.zeros(size)
                self.bias = np.zeros(size[1])  # Initialize bias as zeros
            else:
                self.weights = np.random.randn(*size) * 0.01  # He initialization for stability
                self.bias = np.zeros(size[1])  # Initialize bias as zeros
        else:
            self.weights = weights
            self.bias = np.zeros(weights.shape[1])  # Initialize bias with correct shape

        self.activation_fcn = activation_fcn
        self.activation_derivative = None
        self.output = None
        self.input = None

        # Adam-specific parameters for weights and bias
        self.m_w = np.zeros_like(self.weights)
        self.v_w = np.zeros_like(self.weights)
        self.m_b = np.zeros_like(self.bias)
        self.v_b = np.zeros_like(self.bias)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

    def forward(self, x):
        self.input = x
        self.output = np.dot(x, self.weights) + self.bias  # Include bias
        if self.activation_fcn is not None:
            self.output = self.activation_fcn(self.output)
        return self.output

    def backward(self, output_gradient, learning_rate, use_adam=False):
        if self.activation_derivative is not None:
            output_gradient *= self.activation_derivative(self.output)

        weights_gradient = np.dot(self.input.T, output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0)

        if use_adam:
            # Adam updates for weights
            self.t += 1
            self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * weights_gradient
            self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (weights_gradient ** 2)
            m_hat_w = self.m_w / (1 - self.beta1 ** self.t)
            v_hat_w = self.v_w / (1 - self.beta2 ** self.t)
            self.weights -= learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)

            # Adam updates for bias
            self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * bias_gradient
            self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (bias_gradient ** 2)
            m_hat_b = self.m_b / (1 - self.beta1 ** self.t)
            v_hat_b = self.v_b / (1 - self.beta2 ** self.t)
            self.bias -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
        else:
            # SGD updates
            self.weights -= learning_rate * weights_gradient
            self.bias -= learning_rate * bias_gradient

        return np.dot(output_gradient, self.weights.T)

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_true, y_pred, learning_rate, use_adam=False):
        loss_gradient = cross_entropy_derivative(y_true, y_pred)
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient, learning_rate, use_adam)

    def save_weights(self, save_path):
        weights = [{"weights": layer.weights, "bias": layer.bias} for layer in self.layers]
        with open(save_path, "wb") as f:
            pickle.dump(weights, f)

    def train(self, dataloader, val_dataloader, epochs, learning_rate, num_classes, optimizer="sgd", early_stopping=3,
              plot_interval=1000, plot_format="png"):
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        patience = 0
        use_adam = optimizer.lower() == "adam"

        for epoch in tqdm(range(epochs), desc="Training"):
            epoch_loss = 0
            val_loss = 0

            train_preds, train_labels, train_features = [], [], []
            val_preds, val_labels, val_features = [], [], []

            # Training loop
            for batch_x, batch_y in dataloader.get_batch():
                batch_y_one_hot = one_hot_encode(batch_y, num_classes)
                y_pred = self.forward(batch_x)
                loss = cross_entropy_loss(batch_y_one_hot, y_pred)
                epoch_loss += loss
                self.backward(batch_y_one_hot, y_pred, learning_rate, use_adam)

                # Collect training data for visualization
                train_preds.append(y_pred)
                train_labels.append(batch_y_one_hot)
                train_features.append(batch_x)

            train_loss = epoch_loss / len(dataloader.train_data)
            train_losses.append(train_loss)

            # Validation loop
            for val_x, val_y in val_dataloader.get_batch():
                val_y_one_hot = one_hot_encode(val_y, num_classes)
                val_pred = self.forward(val_x)
                val_loss += cross_entropy_loss(val_y_one_hot, val_pred)

                # Collect validation data for visualization
                val_preds.append(val_pred)
                val_labels.append(val_y_one_hot)
                val_features.append(val_x)

            val_loss /= len(val_dataloader.val_data)
            val_losses.append(val_loss)

            # Log to WandB
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                self.save_weights(f"{args.log_dir}/{args.name}/{epoch}.pkl")
                self.save_weights(f"{args.log_dir}/{args.name}/best_weights.pkl")
            else:
                patience += 1
                if patience >= early_stopping:
                    print(f"Early stopping at epoch {epoch}")

                    train_features_stacked = np.vstack(train_features)
                    train_labels_stacked = np.vstack(train_labels)
                    train_preds_stacked = np.vstack(train_preds)

                    val_features_stacked = np.vstack(val_features)
                    val_labels_stacked = np.vstack(val_labels)
                    val_preds_stacked = np.vstack(val_preds)

                    # Plot training classification
                    self.plot_classification(
                        train_features_stacked, train_labels_stacked, train_preds_stacked,
                        f"Training Classification (Epoch {epoch})",
                        os.path.join(args.log_dir,args.name,f"train_epoch_{epoch}"),
                        format=plot_format
                    )

                    # Plot validation classification
                    self.plot_classification(
                        val_features_stacked, val_labels_stacked, val_preds_stacked,
                        f"Validation Classification (Epoch {epoch})",
                        os.path.join(args.log_dir,args.name,f"val_epoch_{epoch}"),
                        format=plot_format
                    )

                    break


            """            # Plot results every n epochs
            if epoch == epochs - 1:
                train_features_stacked = np.vstack(train_features)
                train_labels_stacked = np.vstack(train_labels)
                train_preds_stacked = np.vstack(train_preds)

                val_features_stacked = np.vstack(val_features)
                val_labels_stacked = np.vstack(val_labels)
                val_preds_stacked = np.vstack(val_preds)

                # Plot training classification
                self.plot_classification(
                    train_features_stacked, train_labels_stacked, train_preds_stacked,
                    f"Training Classification (Epoch {epoch})",
                    os.path.join(args.log_dir, args.name, f"train_epoch_{epoch}"),
                    format=plot_format
                )

                # Plot validation classification
                self.plot_classification(
                    val_features_stacked, val_labels_stacked, val_preds_stacked,
                    f"Validation Classification (Epoch {epoch})",
                    os.path.join(args.log_dir, args.name, f"val_epoch_{epoch}"),
                    format=plot_format
                )"""

    def plot_classification(self, x, y_true, y_pred, title, save_path, format="png", epoch=None):
        """
        Plots the classification results and logs to WandB.
        Points are colored based on their ground truth class.
        Misclassified points are marked with a red cross.
        """

        # Determine ground truth classes and predicted classes
        gt_classes = np.argmax(y_true, axis=1)  # Ground truth class
        pred_classes = np.argmax(y_pred, axis=1)  # Predicted class

        # Define a colormap for the classes
        unique_classes = np.unique(gt_classes)
        class_colors = plt.cm.get_cmap('tab10', len(unique_classes))  # Use tab10 colormap for up to 10 classes

        plt.figure(figsize=(8, 6))
        for i in range(x.shape[0]):
            color = class_colors(gt_classes[i])  # Assign color based on ground truth class
            marker = 'o' if gt_classes[i] == pred_classes[
                i] else 'x'  # Correct points are circles, misclassified are crosses
            plt.scatter(x[i, 0], x[i, 1], color=color, marker=marker, edgecolor="black",
                        s=100 if marker == 'x' else 30)  # Highlight misclassified points

        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True)

        # Save the plot to the specified path
        file_path = f"{save_path}.{format}"
        plt.savefig(file_path, format=format)
        plt.close()

        # Log the plot to WandB
        wandb.log({f"Classification/{title}": wandb.Image(file_path), "epoch": epoch})


def build_network(input_size, num_classes, num_layers, layer_width):
    network = NeuralNetwork()
    network.add(DenseLayer(size=(input_size, layer_width), activation_fcn=activation_fcn, weights_init="rand"))
    network.layers[-1].activation_derivative = activation_derivative

    for _ in range(num_layers - 2):
        network.add(DenseLayer(size=(layer_width, layer_width), activation_fcn=activation_fcn, weights_init="rand"))
        network.layers[-1].activation_derivative = activation_derivative

    network.add(DenseLayer(size=(layer_width, num_classes), activation_fcn=softmax))
    return network

if __name__ == "__main__":
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    path = "data/loader_data"

    if not os.path.exists(f"{args.log_dir}/{args.name}"):
        os.makedirs(f"{args.log_dir}/{args.name}")

    dataloader = CombinedDataloader(path,dataset_name=args.dataset, batch_size=128,mode = "train")
    val_dataloader = CombinedDataloader(path,dataset_name=args.dataset, batch_size=128, mode="val")

    input_size = 2
    num_classes = 6
    epochs = args.epochs
    learning_rate = args.learning_rate
    num_layers = args.num_layers
    layer_width = args.layer_width
    optimizer = args.optimizer

    print(f"Training network with {num_layers} layers, width {layer_width}, activation {args.activation}, "
          f"learning rate {learning_rate}, optimizer {optimizer}")
    network = build_network(input_size, num_classes, num_layers, layer_width)
    network.train(dataloader, val_dataloader, epochs, learning_rate, num_classes, optimizer=optimizer,
                  early_stopping=args.early_stopping)
    wandb.finish()
