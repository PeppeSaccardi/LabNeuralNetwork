# Assignment 3: Deep Neural Network
# Importing the packages we are going to use
import tensorflow as tf
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tensorflow.keras import Model
from tensorflow.keras.layers import Softmax, Flatten
from datetime import datetime

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import MaxPool2D, Dropout
tf.enable_eager_execution()

tf.set_random_seed(0) # set seed for reproducibility

# Loading dataset
data_train = np.loadtxt("train-data.csv",delimiter = ",")
data_test = np.loadtxt("test-data.csv",delimiter = ",")
targets_train = np.array(pd.read_csv("train-target.csv",sep='\t',header = None))
targets_test = np.array(pd.read_csv("test-target.csv",sep='\t', header = None))

# Manipulation needed for computational reason
x = list()
for element_dt in data_train:
  element_dt = element_dt.reshape((16,8))
  M_dt = np.zeros(shape=(16,4))
  element_dt = np.concatenate((element_dt,M_dt), axis=1)
  element_dt = np.concatenate((M_dt,element_dt), axis=1)
  x.append(element_dt)
data_train = np.array(x)
y = list()
for element_test in data_test:
  element_test = element_test.reshape((16,8))
  M_test = np.zeros(shape=(16,4))
  element_test = np.concatenate((element_test,M_test), axis=1)
  element_test = np.concatenate((M_test,element_test), axis=1)
  y.append(element_test)
data_test = np.array(y)

# Convert test to numbers using a particular corrispondence letter/number
targets_train = np.array([ord(element[0]) for element in targets_train])
targets_test = np.array([ord(element[0]) for element in targets_test])
min_val = ord('a')
targets_train = targets_train - min_val
targets_test = targets_test - min_val
n_outputs = 26


# Define metrics
train_loss_metric = tf.keras.metrics.Mean()
train_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
test_loss_metric = tf.keras.metrics.Mean()
test_accuracy_metric = tf.keras.metrics.CategoricalAccuracy()



# Model building

def train_step(images, labels, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_fn(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss_metric(loss)
    train_accuracy_metric(labels, predictions)


def train_loop(epochs, train_ds, model, loss_fn, optimizer):
    epochs_accuracy = []
    epochs_loss = []
    for epoch in range(epochs):
        # reset the metrics for the next epoch
        train_loss_metric.reset_states()
        train_accuracy_metric.reset_states()

        start = datetime.now() # save start time
        for images, labels in train_ds:
            train_step(images, labels, model, loss_fn, optimizer)

        epochs_accuracy.append(train_accuracy_metric.result()*100)
        epochs_loss.append(train_loss_metric.result())

        template = 'Epoch {}, Time {}, Loss: {}, Accuracy: {}'
        print(template.format(epoch+1,
                          datetime.now() - start,
                          train_loss_metric.result(),
                          train_accuracy_metric.result()*100))
    return epochs_accuracy,epochs_loss



def test_step(images, labels, model, loss_fn):
    predictions = model(images, training=False)
    predicted_values = get_predictions(predictions)
    t_loss = loss_fn(labels, predictions)

    test_loss_metric(t_loss)
    test_accuracy_metric(labels, predictions)
    return predicted_values


def get_predictions(predictions):
    values = []
    for element in predictions:
        element = np.array(element)
        max_index = np.argmax(element)
        #convert to characte
        values.append( chr(int(max_index + 97)))
    return values

def test_loop(test_ds, model, loss_fn):
    # reset the metrics for the next epoch
    test_loss_metric.reset_states()
    test_accuracy_metric.reset_states()
    predicted_full = []
     
    for test_images, test_labels in test_ds:
        predicted_values = test_step(test_images, test_labels, model, loss_fn)
        predicted_full = predicted_full + predicted_values

    template = 'Test Loss: {}, Test Accuracy: {}'
    print(template.format(test_loss_metric.result(),
                            test_accuracy_metric.result()*100))
    return test_accuracy_metric.result()*100, predicted_full



# Deep CNN


class DeepConvolutionalNet(Model):
    # convolution layer
    def __init__(self, in_channels, out_channels, size):
        super(DeepConvolutionalNet,self).__init__()
        initial = tf.random.truncated_normal([size, size, in_channels, out_channels], stddev=0.1)
        self.filters = tf.Variable(initial)

    def call(self, x):
        res = tf.nn.conv2d(x, self.filters, 1, padding="SAME")
        return res

class FullyConnected(Model):
    # fully connected layer to perform linear regression
    def __init__(self, input_shape, output_shape):
        super(FullyConnected,self).__init__() # initialize the model
        self.W = tf.Variable(tf.random.truncated_normal([input_shape, output_shape], stddev=0.1)) # declare weights
        self.b = tf.Variable(tf.constant(0.1, shape=[1, output_shape]))  # declare biases
    
    def call(self, x):
        res = tf.matmul(x, self.W) + self.b
        return res

class DeepModel(Model):
    def __init__(self,dropout_val):
        super(DeepModel,self).__init__()              # input shape: (batch,16,16,1)
        self.conv1 = DeepConvolutionalNet(1, 16, 5)   # out shape: (batch,16,16,32)
        self.pool1 = MaxPool2D([2,2])                 # out shape: (batch,8,8,32)
        self.conv2 = DeepConvolutionalNet(16, 32, 5)  # out shape: (batch,8,8,64)
        self.pool2 = MaxPool2D([2,2])                 # out shape: (batch,4,4,32)
        self.conv3 = DeepConvolutionalNet(32, 64, 5)  # out shape: (batch,4,4,64)
      

        self.flatten = Flatten()                      # out shape: (batch,1024)
        self.fc1 = FullyConnected(1024, 1024)         # out shape: (batch,1024)
        self.dropout = Dropout(dropout_val)           # unchanged
        self.fc2 = FullyConnected(1024, 26)           # out shape: (batch,26)
        self.softmax = Softmax()                      # unchanged

    def call(self, x, training=False):
        x = tf.nn.relu(self.conv1(x))
        x = self.pool1(x)
        x = tf.nn.relu(self.conv2(x))
        x = self.pool2(x)
        x = tf.nn.relu(self.conv3(x))
        x = self.flatten(x)
        
        x = tf.nn.relu(self.fc1(x))

        x = self.dropout(x, training=training) # behavior of dropout changes between train and test
    
        x = self.fc2(x)
        prob = self.softmax(x)
    
        return prob
        
        
        

# Using validation set to compare three model with different percentages of dropout


rate = 1e-1
drop_outs = [0.25,0.5,0.75]
accuracy_list = np.zeros(shape=(3))


model_evaluation = True
if model_evaluation == True:
    for i in range(3):
            dropout_val = drop_outs[i]
            # split training data to get training and validation set
            # for model evaluation
            X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(data_train, targets_train, test_size=0.20, random_state=42)
            # convert data to tf.tensor
            x_train = tf.cast(X_train_val, tf.float32)
            x_test = tf.cast(X_test_val, tf.float32)
            # add a fourth dimension
            x_train = x_train[..., tf.newaxis]
            x_test = x_test[..., tf.newaxis]
            # converting to one_hot
            y_train = tf.one_hot(y_train_val.T, n_outputs)
            y_test = tf.one_hot(y_test_val.T, n_outputs)
            # create train and test datasets
            train_ds = tf.data.Dataset.from_tensor_slices(
                (x_train, y_train)).shuffle(10000).batch(200)
            test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(50)

            # create an instance of the model
            network = DeepModel(dropout_val)
            network_loss = tf.keras.losses.CategoricalCrossentropy()
            network_optimizer = tf.keras.optimizers.SGD(learning_rate=rate)
            EPOCHS = 10
            trainloop_accuracy,trainloop_loss = train_loop(EPOCHS, train_ds,  network, network_loss, network_optimizer)
            accuracy_out, testloop_predictions = test_loop(test_ds, network, network_loss)

            accuracy_list[i] = accuracy_out



    # Best drop out 0.25
    best_dropout = drop_outs[accuracy_list.argmax()]

    # convert data to tf.tensor
    x_train = tf.cast(data_train, tf.float32)
    x_test = tf.cast(data_test, tf.float32)
    # add a fourth dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    # converting to one_hot
    y_train = tf.one_hot(targets_train.T, n_outputs)
    y_test = tf.one_hot(targets_test.T, n_outputs)
    # create train and test datasets
    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(10000).batch(200)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(50)

    # create instance of the network
    network = DeepModel(best_dropout)
    network_loss = tf.keras.losses.CategoricalCrossentropy()
    network_optimizer = tf.keras.optimizers.SGD(learning_rate=rate)


# Train
EPOCHS = 30
EA,EL=train_loop(EPOCHS, train_ds,  network, network_loss, network_optimizer)

# Evaluating the accuracy
accuracy_out,predicted_full = test_loop(test_ds, network, network_loss)

# Saving the predictions
np.savetxt("predictions.txt",predicted_full,fmt="%s")



# Plot
plt.grid()
plt.plot(EA)
plt.title("Accuracy Score")
plt.ylim(0.00,110)
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.legend()
plt.savefig("epochs.png",dpi = 300)
