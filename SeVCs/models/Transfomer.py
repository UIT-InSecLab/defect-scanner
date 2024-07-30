from __future__ import print_function
import warnings
from sklearn.metrics import confusion_matrix
from sklearn.utils import compute_class_weight
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import train_test_split
#Transformer
from keras.layers import Input, Dense, Dropout
from keras.layers import MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D
from keras.models import Model
from keras.optimizers import Adamax
from keras.utils import to_categorical

"""
Bidirectional LSTM neural network
Structure consists of two hidden layers and a BLSTM layer
Parameters, as from the VulDeePecker paper:
    Nodes: 300
    Dropout: 0.5
    Optimizer: Adamax
    Batch size: 64
    Epochs: 4
"""
class Transformer:
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

        # inputs
        inputs = Input(shape=(vectors.shape[1], vectors.shape[2]))

        # transformer block 1
        attn_output_1 = MultiHeadAttention(num_heads=8, key_dim=32)(inputs, inputs)
        attn_output_1 = LayerNormalization()(attn_output_1)
        ff_output_1 = Dense(units=32, activation='relu')(attn_output_1)
        ff_output_1 = Dense(units=50)(ff_output_1)
        residual_output_1 = LayerNormalization()(inputs + ff_output_1)

        # transformer block 2
        attn_output_2 = MultiHeadAttention(num_heads=8, key_dim=32)(residual_output_1, residual_output_1)
        attn_output_2 = LayerNormalization()(attn_output_2)
        ff_output_2 = Dense(units=32, activation='relu')(attn_output_2)
        ff_output_2 = Dense(units=50)(ff_output_2)
        residual_output_2 = LayerNormalization()(residual_output_1 + ff_output_2)

        # transformer block 3
        attn_output_3 = MultiHeadAttention(num_heads=8, key_dim=32)(residual_output_2, residual_output_2)
        attn_output_3 = LayerNormalization()(attn_output_3)
        ff_output_3 = Dense(units=32, activation='relu')(attn_output_3)
        ff_output_3 = Dense(units=50)(ff_output_3)
        residual_output_3 = LayerNormalization()(residual_output_2 + ff_output_3)

        # pooling layer
        pooling_output = GlobalAveragePooling1D()(residual_output_3)

        # output layer
        outputs = Dense(units=2, activation='softmax')(pooling_output)

        # compile model
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adamax(learning_rate=0.002)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', 'TruePositives',
                                                                                     'FalsePositives', 'FalseNegatives',
                                                                                     'Precision', 'Recall'])
        self.model = model

    """
    Trains model based on training data
    """
    def train(self):
        self.model.summary()
        self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size, epochs=30,
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
        print("Accuracy is...", values[1])
        recall = tp / (tp + fn)
        print('Recall is...', recall)
        precision = tp / (tp + fp)
        print('Precision is...', precision)
        print('F1 score is...', (2 * precision * recall) / (precision + recall))
