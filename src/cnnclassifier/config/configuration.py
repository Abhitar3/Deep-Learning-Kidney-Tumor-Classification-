import os
from pathlib import Path

from cnnclassifier.constants import *
from cnnclassifier.utils.common import read_yaml, create_directories, save_json
from cnnclassifier.entity.config_entity import DataIngestionConfig
from cnnclassifier.entity.config_entity import prepareBaseModelConfig
from cnnclassifier.entity.config_entity import TrainingConfig
from cnnclassifier.entity.config_entity import EvaluationConfig


class configuartionManager:
    def __init__(self, config_file_path = config_file_path, params_file_path = params_file_path):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = getattr(self.config, "data_ingestion", None)
        if config is None:
            config = getattr(self.config, "data_ing", None)
        if config is None:
            raise AttributeError("Expected 'data_ingestion' or 'data_ing' section in config")

        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )
        return data_ingestion_config

    def get_prepare_base_model_config(self) -> prepareBaseModelConfig:
            config = self.config.prepare_base_model
            create_directories([Path(config.root_dir)])
            params = self.params
            prepare_base_model_config = prepareBaseModelConfig(
                root_dir=Path(config.root_dir),
                base_model_path=Path(config.base_model_path),
                updated_base_model_path=Path(config.updated_base_model_path),
                params_image_size=params.IMAGE_SIZE,
                params_learning_rate=params.LEARNING_RATE,
                params_include_top=params.INCLUDE_TOP,
                params_weights=params.WEIGHTS,
                params_classes=params.CLASSES
            )
            return prepare_base_model_config
    def get_training_config(self) -> TrainingConfig:
            training_config = self.config.training
            prepare_base_model= self.config.prepare_base_model
            params = self.params
            training_data = os.path.join(self.config.data_ingestion.unzip_dir,"KidneyData")
    
            
            training_config = TrainingConfig(
                root_dir=Path(training_config.root_dir),
                trained_model_path=Path(training_config.trained_model_path),
                updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
                training_data=Path(training_data),
                params_epochs=params.EPOCHS,
                params_batch_size=params.BATCH_SIZE,
                params_is_augmentation=params.AUGMENTATION,
                params_image_size=params.IMAGE_SIZE
            )
            return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
            eval_config = EvaluationConfig(
                path_of_model=Path("artifacts/training/trained_model.h5"),
                training_data=Path("artifacts/data_ingestion/unzip/KidneyData"),
                mlflow_uri="https://dagshub.com/Abhitar3/Deep-Learning-Kidney-Tumor-Classification-.mlflow",
                all_params=self.params,
                params_image_size=self.params.IMAGE_SIZE,
                params_batch_size=self.params.BATCH_SIZE
            )
    
            return eval_config