import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


class DeepAxonEngine:
    def __init__(self, sequence_length=10, feature_count=6):
        self.seq_len = sequence_length
        self.feat_count = feature_count
        self.model = self._build_model()
        self.threshold = None

    def _build_model(self):
        model = models.Sequential([
            layers.Conv1D(filters=64, kernel_size=3, activation='relu',
                          padding='same', input_shape=(self.seq_len, self.feat_count)),
            layers.MaxPooling1D(pool_size=2),

            layers.LSTM(100, return_sequences=False),
            layers.Dropout(0.2),

            layers.RepeatVector(self.seq_len),
            layers.LSTM(100, return_sequences=True),
            layers.TimeDistributed(layers.Dense(self.feat_count))
        ])
        model.compile(optimizer='adam', loss='mae')
        return model

    def create_sequences(self, data):
        sequences = []
        for i in range(len(data) - self.seq_len + 1):
            sequences.append(data[i: i + self.seq_len])
        return np.array(sequences)

    def train_model(self, data):
        seq_data = self.create_sequences(data.values)
        self.model.fit(seq_data, seq_data, epochs=15, batch_size=32, verbose=0)

        reconstructions = self.model.predict(seq_data)
        errors = np.mean(np.abs(reconstructions - seq_data), axis=(1, 2))
        self.threshold = np.percentile(errors, 98)  # 2% most extreme are anomalies
        return self.model

    def run_inference(self, data):
        seq_data = self.create_sequences(data.values)
        reconstructions = self.model.predict(seq_data)
        errors = np.mean(np.abs(reconstructions - seq_data), axis=(1, 2))

        predictions = [-1 if e > self.threshold else 1 for e in errors]

        padded_preds = [1] * (self.seq_len - 1) + predictions
        return {"prediction": np.array(padded_preds), "scores": errors}