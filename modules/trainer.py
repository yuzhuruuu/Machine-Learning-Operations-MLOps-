"""trainer.py

Modul training untuk komponen TFX Trainer. Membangun dan melatih model
klasifikasi biner untuk memprediksi risiko diabetes, menggunakan
hyperparameter terbaik hasil komponen Tuner (jika tersedia).
"""

import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

from transform import NUMERICAL_FEATURES, LABEL_KEY, transformed_name
from utils import get_train_eval_dataset

# Hyperparameter default, digunakan apabila komponen Tuner tidak dijalankan
# atau hasil tuning tidak tersedia.
DEFAULT_HYPERPARAMETERS = {
    "units_1": 64,
    "units_2": 32,
    "dropout_rate": 0.3,
    "learning_rate": 1e-3,
}


def _get_hyperparameters(fn_args: FnArgs) -> dict:
    """Mengambil hyperparameter terbaik dari komponen Tuner bila tersedia.

    Args:
        fn_args: argumen training dari komponen Trainer.

    Returns:
        Dictionary berisi hyperparameter yang akan dipakai untuk membangun
        model (hasil tuning bila ada, atau nilai default).
    """
    if getattr(fn_args, "hyperparameters", None):
        hp = kt.HyperParameters.from_config(fn_args.hyperparameters)
        return {
            "units_1": hp.get("units_1"),
            "units_2": hp.get("units_2"),
            "dropout_rate": hp.get("dropout_rate"),
            "learning_rate": hp.get("learning_rate"),
        }
    return DEFAULT_HYPERPARAMETERS


def _get_model(hparams: dict) -> tf.keras.Model:
    """Membangun arsitektur model klasifikasi biner (Keras Functional API).

    Arsitektur:
        - Seluruh 8 fitur numerik yang telah dinormalisasi masuk sebagai
          input dense.
        - 2 hidden layer Dense dengan jumlah unit sesuai hyperparameter,
          aktivasi ReLU, diselingi Dropout untuk mengurangi overfitting.
        - Output layer 1 unit dengan aktivasi sigmoid untuk klasifikasi biner.

    Args:
        hparams: dictionary hyperparameter (units_1, units_2, dropout_rate,
            learning_rate).

    Returns:
        Model Keras yang telah dikompilasi.
    """
    inputs = [
        tf.keras.Input(shape=(1,), name=transformed_name(key))
        for key in NUMERICAL_FEATURES
    ]
    concat = tf.keras.layers.concatenate(inputs)

    dense = tf.keras.layers.Dense(hparams["units_1"], activation="relu")(concat)
    dense = tf.keras.layers.Dropout(hparams["dropout_rate"])(dense)
    dense = tf.keras.layers.Dense(hparams["units_2"], activation="relu")(dense)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(dense)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=hparams["learning_rate"]),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    model.summary()
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Membuat signature function untuk serving model dalam bentuk raw tf.Example."""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY, None)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)
        return model(transformed_features)

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs):
    """Fungsi utama yang dipanggil oleh komponen Trainer TFX.

    Args:
        fn_args: berisi argumen training yang diberikan oleh komponen
            Trainer, termasuk path data train/eval, tf_transform_output,
            dan hyperparameter hasil Tuner (bila ada).
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset, eval_dataset = get_train_eval_dataset(
        fn_args, tf_transform_output, transformed_name(LABEL_KEY), batch_size=32
    )

    hparams = _get_hyperparameters(fn_args)
    print("Hyperparameter yang digunakan untuk training akhir:", hparams)

    model = _get_model(hparams)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq="batch"
    )
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_binary_accuracy", mode="max", patience=5, restore_best_weights=True
    )

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=20,
        callbacks=[tensorboard_callback, early_stopping_callback],
    )

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        )
    }

    tf.saved_model.save(model, fn_args.serving_model_dir, signatures=signatures)
