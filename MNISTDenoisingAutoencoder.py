import numpy as np
from numpy.random import normal
from numpy import clip
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.utils import normalize
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.datasets import mnist
from tensorflow.random import shuffle
from numpy import array
from numpy.random import randn
from tensorflow import convert_to_tensor
from tensorflow import cast
import matplotlib.pyplot as plt
from tensorflow.keras import Model

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print()
print('x train shape: ', x_train.shape)
print('y train shape: ', y_train.shape)
print('x test shape: ', x_test.shape)
print('y test shape: ', y_test.shape)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * normal(loc=0.0, scale=1.0, size=x_test.shape) 

x_train_noisy = clip(x_train_noisy, 0., 1.)
x_test_noisy = clip(x_test_noisy, 0., 1.)
x_noisy = tf.concat((x_train_noisy, x_test_noisy), axis=0)
x_noisy = cast(x_noisy, tf.float32)
x_noisy = x_noisy.numpy()


n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


x_noisy = x_noisy.reshape(-1, 28*28)

# Data preprocessing

x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)
x = tf.concat((x_train, x_test), axis=0)

x = cast(x, tf.float32)

x = x.numpy()

# Encoder

encoding_dim = 8

encoder = Sequential()
encoder.add(Input(shape=(28*28)))

encoder.add(Dense(128, activation='relu'))
encoder.add(Dense(64, activation='relu'))

encoder.add(Dense(encoding_dim, activation='relu'))

# Decoder

decoder = Sequential()
decoder.add(Input(shape=(encoding_dim)))

decoder.add(Dense(64, activation='relu'))
decoder.add(Dense(128, activation='relu'))

decoder.add(Dense(28*28, activation='sigmoid'))

# Autoencoder

model = Sequential()

model.add(encoder)
model.add(decoder)

model.compile(optimizer='adam', loss='binary_crossentropy')

# Train

model.fit(x_noisy, x, epochs=20, batch_size=256, shuffle=True)

# Test

test = x_noisy[0:10]
pred = model.predict(test)
pred = pred.reshape(-1, 28, 28)
test = test.reshape(-1, 28, 28)

for i in range(10):
    plt.imshow(pred[i], cmap=plt.cm.binary)
    plt.title(f'Prediction ({i+1}/10)')
    plt.show()
    
    plt.imshow(test[i], cmap=plt.cm.binary)
    plt.title(f'Actual ({i+1}/10)')
    plt.show()


test = randn(3, encoding_dim)
pred = decoder.predict(test)
for i in range(3):
    plt.imshow(pred[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.title(f'Random input ({i+1}/3)')
    plt.show()
