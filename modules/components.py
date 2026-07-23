"""components.py

Modul ini bertanggung jawab untuk membangun (assemble) seluruh komponen
TFX yang dibutuhkan pada pipeline prediksi risiko diabetes. Dipisah dari
notebook/skrip utama agar pipeline definition mudah dibaca, diuji, dan
digunakan ulang (clean code / separation of concerns).
"""

from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Tuner,
    Trainer,
    Evaluator,
    Pusher,
)
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy,
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
import tensorflow_model_analysis as tfma


def create_example_gen(data_root: str) -> CsvExampleGen:
    """Membuat komponen ExampleGen yang membaca data CSV dan membaginya
    menjadi split train (80%) dan eval (20%)."""
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),
            ]
        )
    )
    return CsvExampleGen(input_base=data_root, output_config=output_config)


def create_statistics_gen(example_gen: CsvExampleGen) -> StatisticsGen:
    """Membuat komponen StatisticsGen dari output ExampleGen."""
    return StatisticsGen(examples=example_gen.outputs["examples"])


def create_schema_gen(statistics_gen: StatisticsGen) -> SchemaGen:
    """Membuat komponen SchemaGen dari output StatisticsGen."""
    return SchemaGen(statistics=statistics_gen.outputs["statistics"])


def create_example_validator(
    statistics_gen: StatisticsGen, schema_gen: SchemaGen
) -> ExampleValidator:
    """Membuat komponen ExampleValidator untuk mendeteksi anomali data."""
    return ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )


def create_transform(
    example_gen: CsvExampleGen, schema_gen: SchemaGen, module_file: str
) -> Transform:
    """Membuat komponen Transform yang menjalankan preprocessing_fn."""
    return Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=module_file,
    )


def create_tuner(transform: Transform, schema_gen: SchemaGen, module_file: str) -> Tuner:
    """Membuat komponen Tuner untuk pencarian hyperparameter otomatis."""
    return Tuner(
        module_file=module_file,
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=100),
        eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=50),
    )


def create_trainer(
    transform: Transform,
    schema_gen: SchemaGen,
    tuner: Tuner,
    module_file: str,
) -> Trainer:
    """Membuat komponen Trainer yang melatih model final menggunakan
    hyperparameter terbaik hasil komponen Tuner."""
    return Trainer(
        module_file=module_file,
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        hyperparameters=tuner.outputs["best_hyperparameters"],
        train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=200),
        eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=100),
    )


def create_model_resolver() -> Resolver:
    """Membuat komponen Resolver untuk mengambil model blessed terakhir
    sebagai baseline pembanding pada tahap Evaluator."""
    return Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("latest_blessed_model_resolver")


def _build_eval_config() -> tfma.EvalConfig:
    """Membangun konfigurasi evaluasi TFMA untuk model prediksi diabetes."""
    return tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="Outcome")],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name="ExampleCount"),
                    tfma.MetricConfig(class_name="AUC"),
                    tfma.MetricConfig(class_name="Precision"),
                    tfma.MetricConfig(class_name="Recall"),
                    tfma.MetricConfig(
                        class_name="BinaryAccuracy",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": 0.6}
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": -1e-3},
                            ),
                        ),
                    ),
                ]
            )
        ],
    )


def create_evaluator(
    example_gen: CsvExampleGen, trainer: Trainer, model_resolver: Resolver
) -> Evaluator:
    """Membuat komponen Evaluator yang mengevaluasi model menggunakan TFMA."""
    return Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=_build_eval_config(),
    )


def create_pusher(
    trainer: Trainer, evaluator: Evaluator, serving_model_dir: str
) -> Pusher:
    """Membuat komponen Pusher yang menyimpan model ke direktori serving
    hanya jika model dinyatakan lolos validasi (blessed)."""
    return Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )


def create_components(
    data_root: str,
    transform_module: str,
    tuner_module: str,
    trainer_module: str,
    serving_model_dir: str,
) -> list:
    """Merangkai seluruh komponen pipeline menjadi satu daftar terurut,
    siap diberikan ke `tfx.orchestration.pipeline.Pipeline`.

    Args:
        data_root: direktori berisi dataset CSV.
        transform_module: path ke modul preprocessing_fn.
        tuner_module: path ke modul tuner_fn.
        trainer_module: path ke modul run_fn.
        serving_model_dir: direktori tujuan Pusher.

    Returns:
        List seluruh komponen TFX secara berurutan.
    """
    example_gen = create_example_gen(data_root)
    statistics_gen = create_statistics_gen(example_gen)
    schema_gen = create_schema_gen(statistics_gen)
    example_validator = create_example_validator(statistics_gen, schema_gen)
    transform = create_transform(example_gen, schema_gen, transform_module)
    tuner = create_tuner(transform, schema_gen, tuner_module)
    trainer = create_trainer(transform, schema_gen, tuner, trainer_module)
    model_resolver = create_model_resolver()
    evaluator = create_evaluator(example_gen, trainer, model_resolver)
    pusher = create_pusher(trainer, evaluator, serving_model_dir)

    return [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    ]
