"""transform.py

Modul preprocessing (Transform) untuk pipeline machine learning prediksi
risiko diabetes. Modul ini digunakan oleh komponen TFX Transform.
"""

import tensorflow as tf
import tensorflow_transform as tft

# Seluruh fitur pada dataset ini bertipe numerik.
NUMERICAL_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

LABEL_KEY = "Outcome"


def transformed_name(key: str) -> str:
    """Mengembalikan nama fitur setelah transformasi (menambahkan suffix '_xf').

    Args:
        key: nama fitur asli.

    Returns:
        Nama fitur dengan suffix '_xf'.
    """
    return key + "_xf"


def _fill_in_missing(x):
    """Mengisi nilai kosong pada sebuah tensor sparse dengan nilai default.

    Args:
        x: tensor sparse atau dense dari sebuah fitur.

    Returns:
        Tensor dense dengan shape [batch_size].
    """
    if isinstance(x, tf.sparse.SparseTensor):
        default_value = "" if x.dtype == tf.string else 0
        x = tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value,
        )
    return tf.squeeze(x, axis=1)


def preprocessing_fn(inputs):
    """Fungsi preprocessing utama yang dipanggil oleh komponen Transform TFX.

    Melakukan normalisasi z-score pada seluruh fitur numerik agar memiliki
    skala yang seragam sebelum masuk ke tahap training.

    Args:
        inputs: dictionary fitur mentah hasil ExampleGen.

    Returns:
        outputs: dictionary fitur yang telah dinormalisasi.
    """
    outputs = {}

    for key in NUMERICAL_FEATURES:
        outputs[transformed_name(key)] = tft.scale_to_z_score(
            _fill_in_missing(inputs[key])
        )

    outputs[transformed_name(LABEL_KEY)] = tf.cast(
        _fill_in_missing(inputs[LABEL_KEY]), tf.int64
    )

    return outputs
