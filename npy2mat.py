import numpy as np
import scipy.io as sio
import os

# Dictionary containing paths for each category
npy_dir = {
    "train_rs": "datasets/UAV-HSI/Train/Training/rs",
    "train_gt": "datasets/UAV-HSI/Train/Training/gt",
    "val_rs": "datasets/UAV-HSI/Train/Validation/rs",
    "val_gt": "datasets/UAV-HSI/Train/Validation/gt",
    "test_rs": "datasets/UAV-HSI/Test/rs",
    "test_gt": "datasets/UAV-HSI/Test/gt",
}

# Loop through each directory in npy_dir
for key, path in npy_dir.items():
    # Initialize an empty list to collect images
    images = []

    # Load and append each .npy file in the directory
    for npy_file in sorted(os.listdir(path)):
        if npy_file.endswith(".npy"):
            image = np.load(os.path.join(path, npy_file))
            images.append(image)

    # Concatenate images along the desired axis (e.g., height axis=0 or width axis=1)
    huge_image = np.concatenate(images, axis=0)  # Use axis=1 to concatenate horizontally

    # Save the concatenated image as a .mat file
    sio.savemat(f"datasets/UAV-HSI/{key}_huge.mat", {key: huge_image})

    print(f"{key}_huge.mat has been created successfully!")
