from cnnclassifier import logger
from cnnclassifier.pipeline.stage1_data_ingestion import DataIngestionPipeline


stage_name = "Data Ingestion stage"
try:
    logger.info(f">>>>> stage {stage_name} started <<<<<")
    obj = DataIngestionPipeline()
    obj.main()
    logger.info(f">>>>> stage {stage_name} completed!<<<<<\n\nx==========x")    
except Exception as e:
    logger.exception(e)
    raise e