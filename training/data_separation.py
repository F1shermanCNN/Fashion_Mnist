import numpy as np
import torchvision.transforms as T
from torchvision.datasets import FashionMNIST
from sklearn.model_selection import train_test_split


def dataset_to_numpy(dataset):
    X = []
    y = []
    
    for img, label in dataset:
        X.append(img.numpy().reshape(-1)) 
        y.append(label)
    
    return np.array(X), np.array(y)


def load_data(data_path='./data'):

    transform = T.Compose([T.ToTensor()])  
    

    train_dataset = FashionMNIST(root=data_path, train=True, download=True, transform=transform)
    test_dataset = FashionMNIST(root=data_path, train=False, download=True, transform=transform)
    

    X_train_full, y_train_full = dataset_to_numpy(train_dataset)
    X_test, y_test = dataset_to_numpy(test_dataset)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=10000, random_state=42)
    
    return X_train, X_val, y_train, y_val, X_test, y_test

