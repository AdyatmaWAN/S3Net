import torch

from sklearn.decomposition import PCA
# import cv2
import scipy.io as sio
# from skimage import data, transform
import os

import sys
import numpy as np

# import matplotlib.pyplot as plt

def loadData(name):
    data_path = os.path.join(os.getcwd(), 'datasets')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    elif name == 'KSC':
        data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
    elif name == 'PCR':
        data = sio.loadmat(os.path.join(data_path, 'Pavia_Center_Right.mat'))['pavia_u_right']
        labels = sio.loadmat(os.path.join(data_path, 'Pavia_Center_Right.mat'))['pavia_gt']
    elif name == 'PCL':
        data = sio.loadmat(os.path.join(data_path, 'Pavia_Center_Left.mat'))['pavia_u_left']
        labels = sio.loadmat(os.path.join(data_path, 'Pavia_Center_Left.mat'))['pavia_gt']
    elif name == 'PD':
        data = sio.loadmat(os.path.join(data_path, 'purdue_hx.mat'))['z']
        labels = sio.loadmat(os.path.join(data_path, 'purdue_gt.mat'))['purdue_gt']
    elif name == 'XA':
        data = sio.loadmat('./datasets/XiongAn.mat')['x']
        labels = sio.loadmat('./datasets/XiongAn_gt.mat')['y']
    elif name == 'HHK':
        data = sio.loadmat('./datasets/ZY_hhk.mat')['ZY_hhk_0628_data']
        labels = sio.loadmat('./datasets/ZY_hhk_gt.mat')['data_gt']
    return data, labels
def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    # indexX = np.arange(0, newX.shape[0], 1)
    pca = PCA(n_components=numComponents, whiten=True)  # whiten是否白化，使得每个特征有相同的方差
    newX = pca.fit_transform(newX)  # 训练pca，让特征在前几维度
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    # indexX = np.reshape(indexX, (X.shape[0], X.shape[1]))
    return newX, pca
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX
def get_idx(X, y):
    data_idx = []
    label_idx = []
    label = []
    whole_idx = []
    for r in range(X.shape[0]):
        for j in range(X.shape[1]):
            whole_idx.append([r, j])
            if y[r,j]!=0:
                data_idx.append([r, j])
                label_idx.append([r, j])
                label.append(y[r, j])
    return data_idx, label_idx, label, whole_idx
def PerClassSplit_idx(data_idx, label_idx, label, seed, perclass):
    np.random.seed(seed)
    x_train_idx = []
    y_train = []
    x_test_idx = []
    y_test = []
    for each_class in range(1, max(label)+1):
        class_number = [i for i in range(len(label)) if label[i] == each_class]
        if perclass < len(class_number)//2:
            train_index = np.random.choice(class_number, perclass, replace=False)
        else:
            train_index = np.random.choice(class_number, len(class_number)//2, replace=False)
        for i in range(len(train_index)):
            index = train_index[i]
            x_train_idx.append(data_idx[index])
            y_train.append(each_class)

        test_index = [i for i in class_number if i not in train_index]

        for i in range(len(test_index)):
            index = test_index[i]
            x_test_idx.append(data_idx[index])
            y_test.append(each_class)
    return x_train_idx, y_train, x_test_idx, y_test
def feature_normalize1(data):
    # mu = np.mean(data, 0)
    # xu = np.std(data, 0)
    # return (data - mu) / xu
    mu = np.mean(data, 0)
    xu = np.std(data, 0)
    return (data - mu) / xu
def feature_normalize2(data):
    return (data - np.min(data, 0)) / (np.max(data,0) - np.min(data,0))


def loadDataUAV(name):
    data_path = os.path.join(os.getcwd(), 'datasets/UAV-HSI')
    if name == 'train':
        data = sio.loadmat(os.path.join(data_path, 'train_rs_huge.mat'))['train_rs']  # Shape: (312, 96, 96, 200)
        labels = sio.loadmat(os.path.join(data_path, 'train_gt_huge.mat'))['train_gt']  # Shape: (312, 96, 96)
    elif name == 'val':
        data = sio.loadmat(os.path.join(data_path, 'val_rs_huge.mat'))['val_rs']
        labels = sio.loadmat(os.path.join(data_path, 'val_gt_huge.mat'))['val_gt']
    else:
        data = sio.loadmat(os.path.join(data_path, 'test_rs_huge.mat'))['test_rs']
        labels = sio.loadmat(os.path.join(data_path, 'test_gt_huge.mat'))['test_gt']
    return data, labels

def get_data(dataset, windowSize1, perclass, K, seed, PCA=False):
    if dataset == "UH":
        X_train_r, y_train_r = loadDataUAV("train")
        X_val_r, y_val_r = loadDataUAV("val")
        X_test_r, y_test_r = loadDataUAV("test")

        X_combined_r = np.concatenate((X_train_r, X_val_r, X_test_r), axis=0)
        y = np.concatenate((y_train_r, y_val_r, y_test_r), axis=0)

        if PCA == True:
            X_combined, _ = applyPCA(X_combined_r, numComponents=K)
        else:
            X_combined = X_combined_r
        X = feature_normalize1(np.asarray(X_combined).astype('float32'))

        del X_combined_r

        data_idx, label_idx, label, whole_idx = get_idx(X, y)

        # Number of rows in the original arrays
        n1 = X_train_r.shape[0]
        n2 = X_val_r.shape[0]
        n3 = X_test_r.shape[0]

        # Separate data_idx and label_idx for X1, X2, X3
        data_idx_train = [idx for idx in data_idx if idx[0] < n1]
        data_idx_val = [idx for idx in data_idx if n1 <= idx[0] < n1 + n2]
        data_idx_test = [idx for idx in data_idx if idx[0] >= n1 + n2]

        label_idx_train = [idx for idx in label_idx if idx[0] < n1]
        label_idx_val = [idx for idx in label_idx if n1 <= idx[0] < n1 + n2]
        label_idx_test = [idx for idx in label_idx if idx[0] >= n1 + n2]

        del X_train_r, X_val_r, X_test_r

        # # Check the lengths or contents if needed
        # print(f"Data indices for train: {len(data_idx_train)}, val: {len(data_idx_val)}, test: {len(data_idx_test)}")
        # print(f"Label indices for train: {len(label_idx_train)}, val: {len(label_idx_val)}, test: {len(label_idx_test)}")

        x_train_idx = data_idx_train
        x_test_idx = data_idx_val

        y_train = [y[idx[0]][idx[1]] for idx in label_idx_train]
        y_test = [y[idx[0]][idx[1]] for idx in label_idx_val]
        #
        # for idx in label_idx_train:
        #     y_train.append(label[idx])
        #
        # for idx in label_idx_val:
        #     y_test.append(label[idx])

        datalist1 = []
        datalist2 = []
        labellist = []
        for order in range(len(y_train)):
            for k in range(len(y_train)):
                y = int(y_train[order] == y_train[k])
                datalist1.append(order)
                datalist2.append(k)
                labellist.append(y)
        margin1 = int((windowSize1 - 1) / 2)
        X = padWithZeros(X, margin=margin1)

        return X, x_train_idx, y_train, x_test_idx, y_test, datalist1, datalist2, labellist, whole_idx

    else:
        X_r, y = loadData(dataset)
        if PCA == True:
            X, _ = applyPCA(X_r, numComponents=K)
        else:
            X = X_r
        X = feature_normalize1(np.asarray(X).astype('float32'))
        data_idx, label_idx, label, whole_idx = get_idx(X, y)
        del X_r
        x_train_idx, y_train, x_test_idx, y_test = PerClassSplit_idx(data_idx, label_idx, label, seed, perclass)

        datalist1 = []
        datalist2 = []
        labellist = []
        for order in range(len(y_train)):
            for k in range(len(y_train)):
                y = int(y_train[order] == y_train[k])
                datalist1.append(order)
                datalist2.append(k)
                labellist.append(y)
        margin1 = int((windowSize1 - 1) / 2)
        X = padWithZeros(X, margin=margin1)
        # sys.exit()

        return X, x_train_idx, y_train, x_test_idx, y_test, datalist1, datalist2, labellist, whole_idx
