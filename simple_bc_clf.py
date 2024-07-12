import glob
import joblib

import numpy as np
import pandas as pd
import pickle as pkl

from glob import glob
from tqdm import tqdm
from xgboost import XGBClassifier
from scipy.signal import find_peaks


def featurizer(clf, img):
    """ img is H, W, C """
    data = img[..., 1:].ravel()
    counts, bins = np.histogram(data, bins=40, density=True)
    n_peaks = len(find_peaks(counts, height=0.5)[0])    
    
    feature = np.array([np.mean(data), np.std(data), n_peaks, *img.shape[:-1]])
    is_big_bc = clf.predict(feature.reshape(1, -1))

    return is_big_bc

if __name__ == "__main__":

    with open('bloodcell_datalabels.pkl', 'rb') as f:
        bc_dataset = pkl.load(f)

    split = 'hidden'
    gt_bc = {i[0].split('/')[-1].split('.')[0]:int(i[2]) for i in bc_dataset[split]}


    img_files = glob(f'/data/user-data/rdilip/cellSAM/dataset/{split}/neurips_fixed/*.X.npy')


    train_model = False
    if train_model:
        bc_data = pkl.load(open('bloodcell_datalabels.pkl', 'rb'))
        dataset = bc_data['train'] + bc_data['test_public'] + bc_data['val']
        hidden_dataset = bc_data['hidden']

        X = [el[1] for el in dataset]
        y = [el[2] for el in dataset]
        Xtest = [el[1] for el in hidden_dataset]
        ytest = [el[2] for el in hidden_dataset]

        ratio = 0.6
        bst = XGBClassifier(n_estimators=3, max_depth=5, learning_rate=1, objective='binary:logistic', scale_pos_weight=0.6,
                            alpha=3, reg_lambda=3)
        bst.fit(X, y)
        preds = bst.predict(X)
        print(np.mean(preds == y))
        preds = bst.predict(Xtest)
        print(np.mean(preds == ytest))

    else:

        clf = joblib.load('./saved_models/new_classifier.pkl')

        correct = 0
        for img in tqdm(img_files):
            base = img.split('/')[-1].split('.')[0]
            im = np.load(img).transpose((1, 2, 0))
            bc_tf = featurizer(clf, im)[0]

            gt = gt_bc[base]

            if bc_tf == gt:
                correct += 1

        print(f'acc {correct/len(img_files)}')