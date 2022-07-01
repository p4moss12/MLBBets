# https://stackoverflow.com/questions/44164749/how-does-keras-handle-multilabel-classification  # HUGE HELP
import typing
import tensorflow as tf
import tensorflow_addons as tfa
# For code completion
from tensorflow import keras

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np
import get_data
import random
import pickle

UNIT_SIZE = 10
BOOK = 'Bovada'
MARKETS = {398: '1H OU',
           402: 'OU',
           91: '1H ML',
           83: 'ML',
           401: 'Point Spread'}
START_DATE = datetime.strptime('2022-04-08', '%Y-%m-%d')  # day after opening day
# END_DATE = datetime.strptime('2022-04-14', '%Y-%m-%d')  # date to train up to
END_DATE = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)  # current date
NUM_EPOCHS = 120


def train(load=True):
    # Load Data
    if load:
        data = np.load('data.npy', allow_pickle=True)
        labels = np.load('labels.npy', allow_pickle=True)
    else:
        get_data.get_data(START_DATE, END_DATE, load=load)  # Get data
        data = np.load('data.npy', allow_pickle=True)  # Load data
        labels = np.load('labels.npy', allow_pickle=True)  # Load labels
    #
    # for index, l in enumerate(data):
    #     print(f'{index}: {len(l)}')
    # return

    # print('Last data and label')
    # print(data[-1])
    # print(labels[-1])
    print(f'Shape of data: {data.shape}')
    print(f'Shape of labels: {labels.shape}')

    # Binarize Labels
    binarizer = preprocessing.MultiLabelBinarizer()
    labels = binarizer.fit_transform(labels)
    pickle.dump(binarizer, open('binarizer.pkl', 'wb'))  # save binarizer
    # Print binarizer classes
    # print(f'Binarizer Classes: {binarizer.classes_}')
    # # Print binarized labels
    # print(f'Binarized labels: {labels}')
    # # Print regular labels from binarizer
    # print(f'Corresponding text labels to last label: {binarizer.inverse_transform(labels)[-1]}')

    # Split data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.05, shuffle=True)

    # Input shape and output layer nodes
    input_shape = X_train[0].shape
    n_labels = y_train.shape[1]  # number of binarized output labels
    print(f'Input shape: {input_shape}')
    print(f'n_labels: {n_labels}')
    print(f'Shape of y_train{y_train.shape}')

    model = keras.Sequential([
        keras.Input(shape=input_shape),  # Input, not technically a layer
        keras.layers.Dense(32, activation='relu', name='1st_Hidden_Layer'),  # Hidden Layer
        keras.layers.Dense(128, activation='relu', name='2nd_Hidden_Layer'),  # Hidden Layer
        keras.layers.Dense(512, activation='relu', name='3rd_Hidden_Layer'),  # Hidden Layer
        keras.layers.Dense(128, activation='relu', name='4th_Hidden_Layer'),  # Hidden Layer
        keras.layers.Dense(32, activation='relu', name='5th_Hidden_Layer'),  # Hidden Layer
        keras.layers.Dense(n_labels, activation='sigmoid', name='Output_Layer')  # Output layer, link has sig reasoning
    ])

    print(model.summary())

    # Compile model
    model.compile(optimizer='adam',
                  loss=keras.losses.BinaryCrossentropy(),  # check link for this choice
                  metrics=[tfa.metrics.F1Score(num_classes=n_labels, threshold=0.9)])  # f1 score = precision and recall
    # Metrics: 'accuracy', tfa.metrics.F1Score, keras.metrics.Precision, keras.metrics.Recall
    # Losses: keras.losses.BinaryCrossentropy()
    # Train model
    model.fit(X_train, y_train, epochs=NUM_EPOCHS)

    # Make predictions on test data; use this to evaluate accuracy, not model.evaluate(see link at top of code)
    preds = model.predict(X_test)
    preds[preds >= 0.9] = 1  # set to 1 if output from model is greater than 0.5 (model output uses sigmoid activation)
    preds[preds < 0.9] = 0  # set to 0 otherwise

    # Get test precision(# of labels predicted that are actually true) and display random prediction
    index = random.choice(range(len(preds)))  # get random index
    preds_ib = binarizer.inverse_transform(preds)  # get inverse binarized labels of predictions
    y_test_ib = binarizer.inverse_transform(y_test)  # get inverse binarized labels of test values

    metric = keras.metrics.Precision()
    metric.update_state(y_test, preds)
    print(f'Test Precision: {metric.result()}')

    # Random prediction
    print('Random prediction vs actual:')
    print(f'Raw prediction: {preds[index]}')
    print(f'Inverse binarized prediction: {preds_ib[index]}')
    print(f'Raw actual: {y_test[index]}')
    print(f'Inverse binarized test: {y_test_ib[index]}')

    # Save model
    model.save('saved_model/MLBModel')


def get_accuracy(y_pred, y_true):  # Actually tests the precision of the model
    correct = 0
    total = 0

    for i in range(len(y_pred)):
        correct += sum(el in y_pred[i] for el in y_true[i])
        total += len(y_pred[i])

    if total == 0:
        return 0

    acc = (correct / total) * 100  # return accuracy

    return acc  # return accuracy


def main():
    train()


if __name__ == '__main__':
    main()
