"""
Hybrid CNN + BiLSTM + Attention with two heads (fault classification + risk regression),
plus Monte Carlo Dropout inference. TensorFlow / Keras.

Ablation variants share the SAME heads, dropout and training recipe; only the encoder changes:
  cnn         : Conv1D x2 -> mean pooling over time
  bilstm      : BiLSTM -> mean pooling over time
  cnn_bilstm  : Conv1D x2 -> BiLSTM -> mean pooling          (fixed aggregation)
  full        : Conv1D x2 -> BiLSTM -> additive attention     (learned aggregation)
"""
from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



class AdditiveAttention(layers.Layer):
    """Bahdanau-style attention pooling over time:
        e_t = v^T tanh(W h_t + b);  a = softmax(e);  c = sum_t a_t h_t
    Returns (context, weights) so weights can be inspected."""
    def __init__(self, units=64, **kw):
        super().__init__(**kw)
        self.units = units

    def build(self, input_shape):
        d = input_shape[-1]
        self.W = self.add_weight(name="W", shape=(d, self.units), initializer="glorot_uniform")
        self.b = self.add_weight(name="b", shape=(self.units,), initializer="zeros")
        self.v = self.add_weight(name="v", shape=(self.units, 1), initializer="glorot_uniform")

    def call(self, h):
        e = tf.squeeze(tf.tensordot(tf.tanh(tf.tensordot(h, self.W, 1) + self.b), self.v, 1), -1)
        a = tf.nn.softmax(e, axis=1)                       # (batch, T)
        c = tf.reduce_sum(h * a[..., None], axis=1)        # (batch, d)
        return c, a

    def get_config(self):
        return {**super().get_config(), "units": self.units}


class MCDropout(layers.Dropout):
    """Dropout that can be forced ON at inference (for Monte Carlo Dropout) while every other
    train-only layer (e.g. GaussianNoise augmentation) stays OFF."""
    mc_on = False

    def call(self, x, training=None):
        return super().call(x, training=True if self.mc_on else training)


def set_mc(model, on: bool):
    for l in model.layers:
        if isinstance(l, MCDropout):
            l.mc_on = on


def build_model(T, F, variant="full", conv_filters=64, kernel=3, lstm_units=64,
                dense_units=64, dropout=0.3, lr=1e-3, risk_weight=10.0, noise=0.0, l2=0.0):
    x_in = keras.Input((T, F), name="sensors")
    reg = keras.regularizers.l2(l2) if l2 else None
    h = layers.GaussianNoise(noise)(x_in) if noise else x_in   # train-time sensor-noise augmentation
    if variant in ("cnn", "cnn_bilstm", "full"):
        h = layers.Conv1D(conv_filters, kernel, padding="same", activation="relu", kernel_regularizer=reg)(h)
        h = layers.Conv1D(conv_filters, kernel, padding="same", activation="relu", kernel_regularizer=reg)(h)
        h = MCDropout(dropout)(h)
    if variant in ("bilstm", "cnn_bilstm", "full"):
        h = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, kernel_regularizer=reg))(h)
        h = MCDropout(dropout)(h)
    if variant == "full":
        z, att = AdditiveAttention(64, name="attention")(h)
    else:
        z, att = layers.GlobalAveragePooling1D()(h), None

    z = layers.Dense(dense_units, activation="relu", kernel_regularizer=reg)(z)
    z = MCDropout(dropout)(z)                         # kept ON at inference for MC Dropout
    fault = layers.Dense(1, activation="sigmoid", name="fault")(z)
    risk = layers.Dense(1, activation="sigmoid", name="risk")(z)   # target lives in [0,1]

    model = keras.Model(x_in, {"fault": fault, "risk": risk}, name=variant)
    # Joint objective: L = 1.0 * BCE(fault) + risk_weight * MSE(risk)
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss={"fault": "binary_crossentropy", "risk": "mse"},
                  loss_weights={"fault": 1.0, "risk": risk_weight})
    att_model = keras.Model(x_in, att) if att is not None else None
    return model, att_model


def train(model, data, epochs=60, batch=256, patience=8, seed=0, verbose=0):
    keras.utils.set_random_seed(seed)
    Xtr, yf, yr = data["train"][:3]
    Xva, vf, vr = data["val"][:3]
    cb = [keras.callbacks.EarlyStopping("val_loss", patience=patience, restore_best_weights=True),
          keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=4, min_lr=1e-5)]
    hist = model.fit(Xtr, {"fault": yf, "risk": yr}, validation_data=(Xva, {"fault": vf, "risk": vr}),
                     epochs=epochs, batch_size=batch, callbacks=cb, verbose=verbose, shuffle=True)
    return hist.history


def predict_det(model, X, batch=2048):
    """Standard deterministic prediction (dropout OFF)."""
    p = model.predict(X, batch_size=batch, verbose=0)
    return p["fault"].ravel(), p["risk"].ravel()


def predict_mc(model, X, passes=50, batch=4096):
    """Monte Carlo Dropout: `passes` stochastic forward passes with dropout ON.
    Returns mean and std (across passes) for both heads, plus raw samples."""
    set_mc(model, True)
    pf, pr = [], []
    for _ in range(passes):
        f_s, r_s = [], []
        for i in range(0, len(X), batch):
            out = model(X[i:i + batch], training=False)   # dropout forced on via set_mc
            f_s.append(out["fault"].numpy().ravel()); r_s.append(out["risk"].numpy().ravel())
        pf.append(np.concatenate(f_s)); pr.append(np.concatenate(r_s))
    set_mc(model, False)
    pf, pr = np.stack(pf), np.stack(pr)                   # (passes, N)
    return {"fault_mean": pf.mean(0), "fault_std": pf.std(0, ddof=1),
            "risk_mean": pr.mean(0), "risk_std": pr.std(0, ddof=1),
            "fault_samples": pf, "risk_samples": pr}
