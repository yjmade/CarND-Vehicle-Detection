# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(1, os.path.dirname(__file__))

import cv2
import glob
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import features

from utils import log_time  # , print_loop


# TODO: Tweak these parameters and see how the results change.
colorspace = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)
hist_bins = 32


MODEL_PATH = os.path.join(os.path.dirname(__file__), "svc.pickle")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def imread(path):
    bgr = cv2.imread(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def extract_feature(img):
    hog_features = features.hog_feature(img, cspace=colorspace, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
    spatial_features = features.bin_spatial(img, size=spatial_size)
    hist_features = features.color_hist(img, nbins=hist_bins)
    return np.hstack((spatial_features, hist_features, hog_features))


def train_model():
    # Divide up into cars and notcars
    # images = glob.iglob('*.jpeg', recursive=True)
    cars = glob.iglob(os.path.join(DATA_PATH, "vehicles", "**", "*.png"), recursive=True)
    notcars = glob.iglob(os.path.join(DATA_PATH, "non-vehicles", "**", "*.png"), recursive=True)
    # for image in images:
    #     if 'image' in image or 'extra' in image:
    #         notcars.append(image)
    #     else:
    #         cars.append(image)

    with log_time("extract car features"):
        car_features = [
            extract_feature(imread(img_path))
            for img_path in cars
        ]
    with log_time("extract noncar features"):
        notcar_features = [
            extract_feature(imread(img_path))
            for img_path in notcars
        ]

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    with log_time("train"):
        svc.fit(X_train, y_train)
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    n_predict = 10
    with log_time("evaluate", n_predict, "items"):
        predicts = svc.predict(X_test[0:n_predict])
    print('My SVC predicts: ', predicts)
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])

    with open(MODEL_PATH, "wb") as f:
        pickle.dump([svc, X_scaler], f)


def get_model():
    with open(MODEL_PATH, "rb") as f:
        clf, X_scaler = pickle.load(f)

    def predict(images):
        features = X_scaler.transform([
            extract_feature(img)
            for img in images
        ])
        return clf.predict(features)

    return predict
