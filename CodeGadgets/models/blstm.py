from __future__ import print_function

import warnings

from keras.utils import to_categorical
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight

warnings.filterwarnings("ignore")

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, LeakyReLU
from keras.optimizers import Adamax

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
"""
Bidirectional LSTM neural network
Structure consists of two hidden layers and a BLSTM layer
Parameters, as from the VulDeePecker paper:
    Nodes: 300
    Dropout: 0.5
    Optimizer: Adamax
    Batch size: 64
    Epochs: 6
"""
class BLSTM:
    def __init__(self, data, name="", batch_size=16):
        vectors = np.stack(data.iloc[:, 0].values)
        labels = data.iloc[:, 1].values
        positive_idxs = np.where(labels == 1)[0]
        negative_idxs = np.where(labels == 0)[0]
        undersampled_negative_idxs = np.random.choice(negative_idxs, len(positive_idxs), replace=False)
        resampled_idxs = np.concatenate([positive_idxs, undersampled_negative_idxs])

        x_train, x_test, y_train, y_test = train_test_split(vectors[resampled_idxs, ], labels[resampled_idxs],
                                                            test_size=0.2, stratify=labels[resampled_idxs])
        self.X_train = x_train
        self.X_test = x_test
        self.y_train = to_categorical(y_train)
        self.y_test = to_categorical(y_test)
        self.name = name
        self.batch_size = batch_size
        self.class_weight = compute_class_weight(class_weight='balanced', classes=[0, 1], y=labels)
        model = Sequential()
        model.add(Bidirectional(LSTM(300), input_shape=(vectors.shape[1], vectors.shape[2])))
        model.add(Dense(300))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(Dense(300))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        # Lower learning rate to prevent divergence
        optimizer = Adamax(learning_rate=0.002)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'TruePositives', 'FalsePositives', 'FalseNegatives', 'Precision', 'Recall'])
        
        plot_model(model, to_file='model_blstm.png', show_shapes=True)
        img = plt.imread('model_blstm.png')
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        self.model = model

    """
    Trains model based on training data
    """
    def train(self):
        self.model.summary()
        self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size, epochs=20,
                       class_weight=dict(enumerate(self.class_weight)))
        self.model.save_weights(self.name + "_model.h5")

    """
    Tests accuracy of model based on test data
    Loads weights from file if no weights are attached to model object
    """
    def test(self):
        self.model.load_weights(self.name + "_model.h5")
        values = self.model.evaluate(self.X_test, self.y_test, batch_size=self.batch_size)
        
        predictions = (self.model.predict(self.X_test, batch_size=self.batch_size)).round()

        tn, fp, fn, tp = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(predictions, axis=1)).ravel()
        recall = tp / (tp + fn)
        print("Accuracy is...", values[1])
        print('Recall is...', recall)
        precision = tp / (tp + fp)
        print('Precision is...', precision)
        print('F1 score is...', (2 * precision * recall) / (precision + recall))