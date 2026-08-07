import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from box import ConfigBox
from mlflow.exceptions import MlflowException
from cnnclassifier import logger
from cnnclassifier.entity.config_entity import EvaluationConfig
from cnnclassifier.utils.common import save_json

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.valid_generator = None
        self.score = None

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    @staticmethod
    def _to_plain_dict(params) -> dict:
        if isinstance(params, ConfigBox):
            return params.to_dict()
        return dict(params)

    @staticmethod
    def _clean_metrics(metrics: dict) -> dict:
        return {
            key: float(value)
            for key, value in metrics.items()
            if value is not None
        }

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator, return_dict=True)
        self.save_score()

    def save_score(self):
        if isinstance(self.score, dict):
            scores = {
                "loss": self.score.get("loss"),
                "accuracy": self.score.get("accuracy")
            }
        elif isinstance(self.score, (list, tuple)):
            scores = {
                "loss": self.score[0] if len(self.score) > 0 else None,
                "accuracy": self.score[1] if len(self.score) > 1 else None
            }
        else:
            scores = {
                "loss": self.score,
                "accuracy": None
            }

        save_json(path=Path("scores.json"), data=scores)

    def _init_dagshub(self):
        try:
            import dagshub

            dagshub.init(
                repo_owner="Abhitar3",
                repo_name="Deep-Learning-Kidney-Tumor-Classification-",
                mlflow=True
            )
            logger.info("DagsHub MLflow tracking initialized")
        except ImportError:
            logger.warning("dagshub is not installed; falling back to mlflow tracking URI")
            mlflow.set_tracking_uri(self.config.mlflow_uri)
        except Exception as e:
            logger.warning(f"DagsHub init failed; falling back to mlflow tracking URI: {e}")
            mlflow.set_tracking_uri(self.config.mlflow_uri)

    def log_into_mlflow(self):
        if self.score is None:
            raise ValueError("Run evaluation() before log_into_mlflow()")
        if self.model is None:
            raise ValueError("Model is not loaded. Run evaluation() first.")

        self._init_dagshub()

        try:
            with mlflow.start_run():
                mlflow.log_params(self._to_plain_dict(self.config.all_params))

                if isinstance(self.score, dict):
                    metrics = {
                        "loss": self.score.get("loss"),
                        "accuracy": self.score.get("accuracy")
                    }
                elif isinstance(self.score, (list, tuple)):
                    metrics = {
                        "loss": self.score[0] if len(self.score) > 0 else None,
                        "accuracy": self.score[1] if len(self.score) > 1 else None
                    }
                else:
                    metrics = {"loss": self.score}

                mlflow.log_metrics(self._clean_metrics(metrics))

                mlflow.keras.log_model(
                    self.model,
                    "model",
                    registered_model_name="VG16Model"
                )
        except MlflowException as e:
            if "403" in str(e):
                raise MlflowException(
                    "DagsHub rejected the MLflow request with 403. "
                    "Log in with `dagshub login`, or set "
                    "MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD "
                    "in the same terminal before running the pipeline."
                ) from e
            raise

    
