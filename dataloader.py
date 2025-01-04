import numpy as np
import os
from glob import glob


class CombinedDataloader:
    def __init__(self, base_path, dataset_name, shuffle=True, batch_size=32,mode = "train"):
        """
        Initialize the dataloader to load and combine data from multiple subdirectories.

        Args:
            base_path (str): Base path containing all subdirectories for tren_data1 and tren_data2.
            dataset_name (str): Either "tren_data1" or "tren_data2" to specify which dataset to load.
            shuffle (bool): Whether to shuffle the data on initialization and between epochs.
            batch_size (int): The size of the batches to return.
        """
        self.base_path = base_path
        self.dataset_name = dataset_name
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.mode = mode

        # Initialize containers for combined data
        self.train_data = []
        self.train_labels = []
        self.val_data = []
        self.val_labels = []
        self.test_data = []
        self.test_labels = []

        self.num_samples = None
        self.batch_index = 0

        # Load and combine data from all relevant subdirectories
        self.load_and_combine_splits()

        # Shuffle training data if specified
        if self.shuffle:
            self.shuffle_train_set()

    def load_and_combine_splits(self):
        """
        Load and combine data from all subdirectories matching the specified dataset name.
        """
        # Find all subdirectories for the specified dataset
        dataset_dirs = sorted(glob(os.path.join(self.base_path, f"{self.dataset_name}*")))

        if not dataset_dirs:
            print(f"No directories found for dataset: {dataset_dirs}")
            raise ValueError(f"No directories found for dataset: {self.dataset_name}")

        for directory in dataset_dirs:
            # Load data and labels from the current subdirectory
            self.train_data.append(np.load(os.path.join(directory, "train_data.npy")))
            self.train_labels.append(np.load(os.path.join(directory, "train_labels.npy")))
            self.val_data.append(np.load(os.path.join(directory, "val_data.npy")))
            self.val_labels.append(np.load(os.path.join(directory, "val_labels.npy")))
            self.test_data.append(np.load(os.path.join(directory, "test_data.npy")))
            self.test_labels.append(np.load(os.path.join(directory, "test_labels.npy")))

        # Combine all loaded data and labels
        self.train_data = np.vstack(self.train_data)
        self.train_labels = np.hstack(self.train_labels)
        self.val_data = np.vstack(self.val_data)
        self.val_labels = np.hstack(self.val_labels)
        self.test_data = np.vstack(self.test_data)
        self.test_labels = np.hstack(self.test_labels)

        # Set the number of training samples
        self.num_samples = len(self.train_data)

    def shuffle_train_set(self):
        """Shuffle the training data and labels."""
        p = np.random.permutation(len(self.train_data))
        self.train_data = self.train_data[p]
        self.train_labels = self.train_labels[p]

    def get_batch(self):
        """
        Get a batch from the specified dataset type.

        Args:
            dataset_type (str): One of "train", "val", or "test".

        Yields:
            tuple: A batch of data and labels.
        """
        if self.mode == "train":
            data = self.train_data
            labels = self.train_labels
        elif self.mode == "val":
            data = self.val_data
            labels = self.val_labels
        elif self.mode == "test":
            data = self.test_data
            labels = self.test_labels


        # Reset batch index for new epoch
        self.batch_index = 0

        while self.batch_index < len(data):
            batch_data = data[self.batch_index:self.batch_index + self.batch_size]
            batch_labels = labels[self.batch_index:self.batch_index + self.batch_size]
            self.batch_index += self.batch_size
            yield batch_data, batch_labels

    def __len__(self):
        """Return the number of samples in the training set."""
        return self.num_samples // self.batch_size


if __name__ == "__main__":
    # Example usage: Load and combine all data from the specified dataset
    base_path = "data/loader_data"  # Replace with the base path of your data
    dataset_name = "tren_data1"  # or "tren_data2"

    dataloader = CombinedDataloader(base_path, dataset_name)

    # Example: Iterate through the first 10 batches of training data
    for i, (data, labels) in enumerate(dataloader.get_batch()):
        print(f"Batch {i + 1}:")
        print("Data:", data)
        print("Labels:", labels)
        if i == 9:  # Stop after 10 batches
            break

    print("Dataloader initialized and data combined successfully.")
