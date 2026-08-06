from cnnclassifier import logger
from cnnclassifier.pipeline.stage1_data_ingestion import DataIngestionPipeline
from cnnclassifier.pipeline.stage2_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnclassifier.pipeline.stage3_model_training import ModelTrainingPipeline



stage_name = "Data Ingestion stage"
try:
    logger.info(f">>>>> stage {stage_name} started <<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f">>>>> stage {stage_name} completed!<<<<<\n\nx==========x")    
except Exception as e:
    logger.exception(e)
    raise e


stage_name = "Prepare Base Model stage"
try:    
    logger.info(f">>>>> stage {stage_name} started <<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {stage_name} completed!<<<<<\n\nx==========x")    
except Exception as e:
    logger.exception(e)
    raise e


stage_name = "Model Training stage"
try:
    logger.info(f">>>>> stage {stage_name} started <<<<<")
    from cnnclassifier.pipeline.stage3_model_training import ModelTrainingPipeline
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {stage_name} completed!<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
