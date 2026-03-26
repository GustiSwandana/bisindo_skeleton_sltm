from __future__ import annotations

from tensorflow import keras


def build_lstm_classifier(
    sequence_length: int,
    feature_dim: int,
    num_classes: int,
    lstm_units: int = 128,
    dense_units: int = 64,
    dropout_rate: float = 0.3,
) -> keras.Model:
    inputs = keras.Input(shape=(sequence_length, feature_dim), name="landmark_sequence")
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(lstm_units, return_sequences=True)
    )(inputs)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Bidirectional(keras.layers.LSTM(lstm_units // 2))(x)
    x = keras.layers.Dense(dense_units, activation="relu")(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="alphabet")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="bisindo_skeleton_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
