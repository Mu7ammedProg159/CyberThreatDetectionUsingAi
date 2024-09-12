from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, auc
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.utils import resample
from sklearn.svm import OneClassSVM
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from functions import preprocessing_data
import torch
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import tensorflow as tf
from joblib import dump, load

def train_all(X_train, y_train, X_val, y_val) :
    # Define the XGBoost classifier
    xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', learning_rate= 0.01, max_depth = 3, reg_alpha = 0.01, reg_lambda = 0.01, num_class=3, seed=42)
    # Train the classifier on the training data
    xgb_classifier.fit(X_train, y_train)


    #training random forest clf
    dt_classifier = tree.DecisionTreeClassifier(criterion='gini', splitter="best", max_leaf_nodes=7, min_samples_leaf=1, min_samples_split=2, max_depth=None, random_state=42)
    dt_classifier = dt_classifier.fit(X_train, y_train)


    np.random.seed(123)

    # Set the default data type explicitly
    tf.keras.backend.set_floatx('float32')

    # Assuming X_train, X_test, y_train, y_test are your data
    # Convert DataFrame objects to numpy arrays
    X_train_array = X_train.values
    X_test_array = X_test.values
    X_val_array = X_val.values  # Convert X_val to numpy array

    # Reshape the input data to match the GRU input shape
    X_train_reshaped = np.reshape(X_train_array, (X_train_array.shape[0], X_train_array.shape[1], 1))
    X_val_reshaped = np.reshape(X_val_array, (X_val_array.shape[0], X_val_array.shape[1], 1))
    X_test_reshaped = X_test_array.reshape(X_test_array.shape[0], X_test_array.shape[1], 1)

    # Define the model
    model = Sequential([
        GRU(units=64, input_shape=(X_train_reshaped.shape[1], 1)),
        Dense(units=1, activation='sigmoid')
    ])

    # Compile the model with a specific learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    gru_model = model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_val_reshaped, y_val))