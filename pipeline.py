"""pipeline.py

Skrip utama untuk menjalankan machine learning pipeline prediksi risiko
diabetes menggunakan TensorFlow Extended (TFX), dieksekusi dengan
**Apache Beam** sebagai pipeline orchestrator (BeamDagRunner).

Cara menjalankan:
    python pipeline.py
"""

import os
import sys

from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_ROOT = os.path.join(BASE_DIR, "modules")
sys.path.insert(0, MODULE_ROOT)

from components import create_components  # noqa: E402  (needs sys.path insert above)

PIPELINE_NAME = "yusriiii-pipeline"

DATA_ROOT = os.path.join(BASE_DIR, "data")
PIPELINE_ROOT = os.path.join(BASE_DIR, PIPELINE_NAME)
SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, "serving_model")
METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata.sqlite")

TRANSFORM_MODULE_FILE = os.path.join(MODULE_ROOT, "transform.py")
TUNER_MODULE_FILE = os.path.join(MODULE_ROOT, "tuner.py")
TRAINER_MODULE_FILE = os.path.join(MODULE_ROOT, "trainer.py")


def init_local_pipeline(components: list, pipeline_root: str) -> pipeline.Pipeline:
    """Membungkus daftar komponen menjadi objek TFX Pipeline yang siap
    dijalankan menggunakan Apache Beam sebagai orchestrator.

    Args:
        components: daftar komponen TFX yang sudah dirangkai berurutan.
        pipeline_root: direktori root untuk menyimpan seluruh artefak
            pipeline (folder <username_dicoding>-pipeline).

    Returns:
        Objek tfx.orchestration.pipeline.Pipeline.
    """
    beam_pipeline_args = [
        "--direct_running_mode=multi_processing",
        "--direct_num_workers=0",
    ]

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            METADATA_PATH
        ),
        beam_pipeline_args=beam_pipeline_args,
    )


def run_pipeline():
    """Menjalankan seluruh pipeline end-to-end menggunakan BeamDagRunner."""
    logging.set_verbosity(logging.INFO)

    components = create_components(
        data_root=DATA_ROOT,
        transform_module=TRANSFORM_MODULE_FILE,
        tuner_module=TUNER_MODULE_FILE,
        trainer_module=TRAINER_MODULE_FILE,
        serving_model_dir=SERVING_MODEL_DIR,
    )

    tfx_pipeline = init_local_pipeline(components, PIPELINE_ROOT)
    BeamDagRunner().run(tfx_pipeline)


if __name__ == "__main__":
    run_pipeline()
