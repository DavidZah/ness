import os

import matplotlib.pyplot as plt
import numpy as np

def load_data(path):
    data_lst = []
    labels_lst = []
    with open(path,"r") as f:
        data = f.readlines()
        for line in data:
            line = line.strip().split("  ")
            data_lst.append([float(x) for x in line[0:-1]])
            labels_lst.append(float(line[-1].strip()))

    data = np.array(data_lst)
    labels = np.array(labels_lst)

    return data, labels

def randomize_and_split_to_train_val_test(data,labels):
    p = np.random.permutation(len(data))
    data = data[p]
    labels = labels[p]

    train_size = int(len(data) * 0.8)
    val_size = int(len(data) * 0.1)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    train_labels = labels[:train_size]
    val_labels = labels[train_size:train_size + val_size]
    test_labels = labels[train_size + val_size:]

    return train_data, train_labels, val_data, val_labels, test_data, test_labels




if __name__ == "__main__":
    path = "/Users/zah/PycharmProjects/ness_1/data/tren_data"
    save_path = "/Users/zah/PycharmProjects/ness_1/data/loader_data"
    data_lst = os.listdir(path)

    for i in data_lst:
        data, labels = load_data(os.path.join(path, i))
        train_data, train_labels, val_data, val_labels, test_data, test_labels = randomize_and_split_to_train_val_test(data,labels)
        #this is two layer i need to created nested folder
        os.makedirs(os.path.join(save_path, i.split(".")[0]))
        np.save(os.path.join(save_path, i.split(".")[0], "train_data.npy"), train_data)
        np.save(os.path.join(save_path, i.split(".")[0], "train_labels.npy"), train_labels)
        np.save(os.path.join(save_path, i.split(".")[0], "val_data.npy"), val_data)
        np.save(os.path.join(save_path, i.split(".")[0], "val_labels.npy"), val_labels)
        np.save(os.path.join(save_path, i.split(".")[0], "test_data.npy"), test_data)
        np.save(os.path.join(save_path, i.split(".")[0], "test_labels.npy"), test_labels)

    print("done")