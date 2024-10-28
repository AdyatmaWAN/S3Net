import scipy.io as sio
import os

from six import print_


def main():
    data_path = os.path.join(os.getcwd(), 'datasets')
    data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
    labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']

    print("Salinas dataset")
    # print(data)
    print(data.shape)
    # print()
    # print(labels)
    print(labels.shape)
    print()


    data = sio.loadmat("datasets/UAV-HSI/train_rs_huge.mat")
    data = data["train_rs"]
    labels = sio.loadmat("datasets/UAV-HSI/train_gt_huge.mat")
    labels = labels["train_gt"]
    print("UAV-HSI Dataset")
    # print(data)
    print(data.shape)
    # print()
    # print(labels)
    print(labels.shape)
    print()


    pass

if __name__ == "__main__":
    main()