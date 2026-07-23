"""tuner.py

Modul hyperparameter tuning untuk komponen TFX Tuner. Menggunakan
KerasTuner (RandomSearch) untuk mencari kombinasi hyperparameter terbaik
sebelum model final dilatih oleh komponen Trainer.
"""

from typing import NamedTuple, Any, Dict

import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

from transform import NUMERICAL_FEATURES, LABEL_KEY, transformed_name
from utils import get_train_eval_dataset

TunerFnResult = NamedTuple(
    "TunerFnResult", [("tuner", Any), ("fit_kwargs", Dict[str, Any])]
)


def _build_model(hp: kt.HyperParameters) -> tf.keras.Model:
    """Membangun model dengan hyperparameter yang dapat di-tuning.

    Hyperparameter yang di-tuning:
        - units_1: jumlah unit pada hidden layer pertama.
        - units_2: jumlah unit pada hidden layer kedua.
        - dropout_rate: rasio dropout.
        - learning_rate: laju pembelajaran optimizer Adam.

    Args:
        hp: objek HyperParameters dari KerasTuner.

    Returns:
        Model Keras yang telah dikompilasi.
    """
    inputs = [
        tf.keras.Input(shape=(1,), name=transformed_name(key))
        for key in NUMERICAL_FEATURES
    ]
    concat = tf.keras.layers.concatenate(inputs)

    units_1 = hp.Int("units_1", min_value=16, max_value=128, step=16)
    units_2 = hp.Int("units_2", min_value=8, max_value=64, step=8)
    dropout_rate = hp.Float("dropout_rate", min_value=0.1, max_value=0.5, step=0.1)
    learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    dense = tf.keras.layers.Dense(units_1, activation="relu")(concat)
    dense = tf.keras.layers.Dropout(dropout_rate)(dense)
    dense = tf.keras.layers.Dense(units_2, activation="relu")(dense)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(dense)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")],
    )
    return model


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Fungsi utama yang dipanggil oleh komponen Tuner TFX.

    Args:
        fn_args: argumen tuning yang diberikan oleh komponen Tuner, termasuk
            path data train/eval dan tf_transform_output.

    Returns:
        TunerFnResult berisi objek tuner KerasTuner dan fit_kwargs.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset, eval_dataset = get_train_eval_dataset(
        fn_args, tf_transform_output, transformed_name(LABEL_KEY), batch_size=32
    )

    tuner = kt.RandomSearch(
        _build_model,
        objective=kt.Objective("val_binary_accuracy", direction="max"),
        max_trials=8,
        directory=fn_args.working_dir,
        project_name="diabetes_tuning",
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
            "epochs": 5,
        },
    )
