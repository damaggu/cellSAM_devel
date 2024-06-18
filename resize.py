import matplotlib.pyplot as plt

files = "/home/markus/PycharmProjects/cellSAM_devel/results_hidden_256_100_100_tuning_tryCatch_2/inferences/"
save_path = "/home/markus/PycharmProjects/cellSAM_devel/results_hidden_256_100_100_tuning_tryCatch_2/inferences_reshaped/"
save_path_png = "/home/markus/PycharmProjects/cellSAM_devel/results_hidden_256_100_100_tuning_tryCatch_2/inferences_reshaped_png/"

import os
import imageio.v3 as iio

if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists(save_path_png):
    os.makedirs(save_path_png)

all_files = os.listdir(files)

for file in all_files:

    labels = iio.imread(files + file)

    # gt_path = "/home/markus/PycharmProjects/cellSAM_devel/neurips_tuning/Tuning/images/" + file.split('.')[0]
    gt_path = "/home/markus/PycharmProjects/cellSAM_devel/neurips_hidden/Testing/Hidden/images/" + file.split('.')[0]
    endings = ['.tif', '.tiff', '.png', '.bmp']
    for end in endings:
        if os.path.exists(gt_path + end):
            gt_path = gt_path + end
            break

    gt_mask = iio.imread(gt_path)
    if gt_mask.shape[0] < 512:
        # remove padding
        padding = 512 - gt_mask.shape[0]
        labels = labels[padding // 2:-padding // 2, :]
    if gt_mask.shape[1] < 512:
        padding = 512 - gt_mask.shape[1]
        labels = labels[:, padding // 2:-padding // 2]

    iio.imwrite(save_path + file, labels)

    # save as png
    plt.imshow(labels)
    plt.title(file)
    plt.savefig(save_path_png + file.split('.')[0] + '.png')
    plt.close()

