""" Multilayer Perceptron"""
from __future__ import print_function
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras import layers
from keras import optimizers
import os
import glob
import tensorflow as tf

# from keras.utils import plot_model
import keras.backend as K
from keras.callbacks import ModelCheckpoint
# from tensorflow.contrib import predictor
# import os 

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class kerasChessNetwork:
    def __init__(self, weightPath = None):

        # self.mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
        self.trainingDirec = 'training'
        

        # Parameters
        self.learning_rate = 0.001
        self.training_epochs = 500
        self.batch_size = 256
        self.display_step = 1

        # Network Parameters
        self.n_neurons_1 = 2048
        self.n_neurons_2 = 2048
        self.n_neurons_3 = 2048
        # self.n_neurons_4 = 2048

        self.n_boards = 768

        self.n_classes = 1 # Total result

        self.X_test = None
        self.X_train = None
        self.y_test = None
        self.y_train = None

        self.weightPath = weightPath

        self.vfunc = np.vectorize(self.minMax)

        self.createModel()
        self.graph = tf.get_default_graph()


    def minMax(self, num):
        min_bound = -3000
        max_bound = 3000
        if num < min_bound:
            return 0.
        if num > max_bound:
            return 1.
        else:
            return (num+3000)/6000

    def readInData(self):
        # data = pd.read_csv(filepath, header=None)
        path = self.trainingDirec # use your path
        # all_files = glob.glob(path + "/*.csv")

        all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent

        df_from_each_file = (pd.read_csv(f, header=None) for f in all_files)
        data   = pd.concat(df_from_each_file, ignore_index=True, axis=0)

        # print (data)

        # Dimensions of dataset
        n = data.shape[0]
        p = data.shape[1]

        print ("Number of total boards: %4.2f" % n)


        # # Training and test data
        # train_start = 0
        # train_end = int(np.floor(0.8*n))

        # test_start = train_end + 1
        # test_end = n

        data_train = data.sample(frac=0.8)
        data_test = data.loc[~data.index.isin(data_train.index)]

        # Make data a np.array
        self.data_train = data_train.values
        self.data_test = data_test.values

        # data_train = data[np.arange(train_start, train_end), :]
        # data_test = data[np.arange(test_start, test_end), :]


        # Build X and y
        self.X_train = self.data_train[:, 1:]
        self.y_train = self.data_train[:, 0]

        self.X_test = self.data_test[:, 1:]
        self.y_test = self.data_test[:, 0]

        # print (min(self.y_train))
        # print (max(self.y_train))
        # print (self.y_train)
        #scale data
        
        self.y_train = self.vfunc(self.y_train)
        self.y_test = self.vfunc(self.y_test)
        # self.y_train = sklearn.preprocessing.minmax_scale(self.y_train)
        # self.y_test = sklearn.preprocessing.minmax_scale(self.y_test)

        # print (self._test)
        # print (self.y_train)

        self.data_summary()

        # Number of boards in training data
        # self.n_boards = self.X_train.shape[1]
        # print (self.n_boards)

    def data_summary(self):
        """Summarize current state of dataset"""
        print('Train board shape:', self.X_train.shape)
        print('Train labels shape:', self.y_train.shape)
        print('Test board shape:', self.X_test.shape)
        print('Test labels shape:', self.y_test.shape)
        print('Train labels:', self.y_train)
        print('Test labels:', self.y_test)

    def createModel(self):

        self.model = Sequential()
        self.model.add(Dense(self.n_neurons_1, input_dim=self.n_boards, activation='elu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(Dense(self.n_neurons_2, activation='elu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(Dense(self.n_neurons_3, activation='elu'))
        self.model.add(layers.BatchNormalization())
        # self.model.add(Dense(self.n_neurons_3, activation='elu'))
        self.model.add(Dense(1, activation='elu'))

        sgd = optimizers.SGD(lr=0.001, decay=1e-8, momentum=0.7, nesterov=True)
        # Compile model
        self.model.compile(loss='mean_squared_error', optimizer=sgd, metrics = ['mse'])
        # self.model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics = ['mse'])

        if self.weightPath:
            self.model.load_weights(self.weightPath)
            print ("Loaded Weights")

        # plot_model(self.model, to_file='model.png')


        return self.model

    def trainModel(self):

        self.readInData()

        filepath="model_keras/weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False)
        callbacks_list = [checkpoint]

        # Fit the model
        print (self.X_train.shape)
        print (self.y_train.shape)
        self.model.fit(self.X_train, self.y_train, epochs=120, batch_size=256, validation_split = 0.1, callbacks=callbacks_list)
        # evaluate the model
        scores = self.model.evaluate(self.X_test, self.y_test)
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))
        self.model.save_weights('model_keras/keras_net.h5')

    def testModel(self, dataset_path):
        evaluate_pd = pd.read_csv(dataset_path, header=None)
        # print(evaluate_pd)
        evaluate_np = evaluate_pd.values
        # print(evaluate_np)
        X_eval = evaluate_np[:, 1:]
        y_eval = evaluate_np[:, 0]
        y_eval = y_eval.reshape((-1,1))
        y_eval = self.vfunc(y_eval)
        # print (y_eval)
        result = self.model.predict(X_eval)
        print (result.reshape((-1,1)))
        scores = self.model.evaluate(X_eval, y_eval)
        print(scores)
        print("\n%s: %.5f" % (self.model.metrics_names[1], scores[1]))

    def retrieveBoardValueNeuralNet(self, boardInput):
        test = np.array([boardInput])
        with self.graph.as_default():
            result = self.model.predict(test)
            return result[0][0]  


# def rmse (y_true, y_pred):
#     return K.sqrt(K.mean(K.square(y_pred -y_true)))

if __name__ == '__main__':
    # chessNet = chessNetwork('model/net.ckpt')
    # chessNet = kerasChessNetwork('model_keras/after_training_99_weights.best.hdf5')
    chessNet = kerasChessNetwork('model_keras/weights.best.hdf5')
    
    chessNet.trainModel()
    # chessNet.testModel('training/newerish/training18.csv')
    # chessNet.createModel()
    # chessNet.readInData()
    # 

    # chessNet.loadModel("model/save_net14.ckpt")
    
    # a = chessNet.retrieveBoardValueNeuralNet([0,-1,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
    # b = chessNet.retrieveBoardValueNeuralNet([0,-1,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])

    # print (a)
    # print (b)

    # chessNet.closeSession()



