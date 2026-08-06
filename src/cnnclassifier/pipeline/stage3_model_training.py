from cnnclassifier.config.configuration import configuartionManager
from cnnclassifier.components.model_training import Training    
from cnnclassifier import logger



STAGE_NAME = "Model Training Stage"


class ModelTrainingPipeline:
    def __init__(self):
        pass


    def main(self):
        config = configuartionManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()

if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed!<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
    