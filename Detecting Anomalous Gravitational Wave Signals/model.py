import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.special import expit 


class Model:
    def __init__(self):
        super().__init__()

    def encoder(self, inputs):
        x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(inputs)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(16, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Conv1D(16, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(400, activation="relu")(x)
        return x

    def decoder(self, inputs, original_shape):
        x = layers.Dense(1600, activation="relu")(inputs)
        x = layers.Reshape((25, 64))(x)
        x = layers.Conv1D(16, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.UpSampling1D(size=2)(x)
        x = layers.Conv1D(16, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.UpSampling1D(size=2)(x)
        x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.UpSampling1D(size=2)(x)
        x = layers.Conv1D(2, kernel_size=3, padding="same", activation="sigmoid")(x)
        return x

    def build_model(self, input_shape):
        inputs = keras.Input(shape=input_shape)

        # Encoder
        encoded = self.encoder(inputs)

        # Decoder
        decoded = self.decoder(encoded, input_shape)

        # Autoencoder Model
        self.ae = keras.Model(inputs, decoded)
        self.ae.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=1e-5))

    def predict(self, X, batch_size=32):
        reconstruction_error = np.mean((self.ae.predict(X, batch_size=batch_size) - X) ** 2, axis=(1, 2))
        probabilities = expit(-reconstruction_error)
        return probabilities

    def __call__(self, inputs, batch_size=64):
        return self.ae.predict(inputs, batch_size=batch_size)

    def save(self, path):
        self.ae.save(os.path.join(os.path.dirname(__file__), 'model.keras'))

    def load(self):
        self.ae = keras.models.load_model(os.path.join(os.path.dirname(__file__), 'model.keras'))


    def fit(self, x_train, **kwargs):
        initial_learning_rate = 1e-5
        decay_steps = 10000  # Adjust as needed
        decay_rate = 0.96  # Adjust as needed

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.ae.compile(loss="mse", optimizer=optimizer)

        history = self.ae.fit(x_train, x_train, **kwargs)
        return history
