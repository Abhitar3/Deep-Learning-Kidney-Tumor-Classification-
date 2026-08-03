from cnnclassifier.constants import *
from cnnclassifier.utils.common import read_yaml, create_directories
from cnnclassifier.entity.config_entity import DataIngestionConfig
from cnnclassifier.entity.config_entity import prepareBaseModelConfig

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
