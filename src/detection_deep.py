import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import logging


class DeepAxonEngine:
    def __init__(self, sequence_length=10, feature_count=6):
        self.seq_len = sequence_length
        self.feat_count = feature_count
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential([
            layers.Conv1D(filters=64, kernel_size=3, activation='relu',
                          input_shape=(self.seq_len, self.feat_count)),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),

            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.2),

            layers.RepeatVector(self.seq_len),
            layers.LSTM(50, return_sequences=True),
            layers.TimeDistributed(layers.Dense(self.feat_count))
        ])
        model.compile(optimizer='adam', loss='mae')
        return model

    def prepare_sequences(self, data):
        sequences = []
        for i in range(len(data) - self.seq_len):
            sequences.append(data[i: i + self.seq_len])
        return np.array(sequences)

    def train(self, data, epochs=20, batch_size=32):
        sequences = self.prepare_sequences(data)
        self.model.fit(sequences, sequences, epochs=epochs,
                       batch_size=batch_size, validation_split=0.1, verbose=1)

    def get_anomaly_scores(self, data):
        sequences = self.prepare_sequences(data)
        predictions = self.model.predict(sequences)
        mse = np.mean(np.abs(sequences - predictions), axis=(1, 2))
        return mse