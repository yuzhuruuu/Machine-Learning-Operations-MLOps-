"""utils.py

Fungsi-fungsi utilitas yang dipakai bersama oleh `trainer.py` dan
`tuner.py`, dipisahkan ke modul tersendiri untuk menghindari duplikasi
kode (prinsip *clean code* / DRY - Don't Repeat Yourself).
"""

import tensorflow as tf


def gzip_reader_fn(filenames):
    """Loader untuk membaca file TFRecord terkompresi GZIP.

    Args:
        filenames: daftar path file TFRecord.

    Returns:
        tf.data.TFRecordDataset yang membaca file terkompresi GZIP.
    """
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(file_pattern, tf_transform_output, label_key, batch_size=32):
    """Membuat tf.data.Dataset yang siap dipakai untuk training/tuning/evaluasi.

    Args:
        file_pattern: pola path untuk file input (hasil Transform).
        tf_transform_output: TFTransformOutput hasil dari komponen Transform.
        label_key: nama kolom label (setelah transformasi).
        batch_size: jumlah sampel per batch.

    Returns:
        tf.data.Dataset yang menghasilkan pasangan (fitur, label).
    """
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    return tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=label_key,
    )


def get_train_eval_dataset(fn_args, tf_transform_output, label_key, batch_size=32):
    """Membuat dataset train dan eval sekaligus dari `FnArgs`.

    Fungsi ini dipakai bersama oleh `trainer.py` dan `tuner.py` agar logika
    pembuatan dataset train/eval tidak diduplikasi di kedua modul tersebut.

    Args:
        fn_args: objek FnArgs yang diberikan oleh komponen Trainer/Tuner,
            berisi path file train/eval.
        tf_transform_output: TFTransformOutput hasil komponen Transform.
        label_key: nama kolom label (setelah transformasi).
        batch_size: jumlah sampel per batch.

    Returns:
        Tuple (train_dataset, eval_dataset).
    """
    train_dataset = input_fn(
        fn_args.train_files, tf_transform_output, label_key, batch_size=batch_size
    )
    eval_dataset = input_fn(
        fn_args.eval_files, tf_transform_output, label_key, batch_size=batch_size
    )
    return train_dataset, eval_dataset
