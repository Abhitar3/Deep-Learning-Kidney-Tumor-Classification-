import os
import zipfile
import gdown
from cnnclassifier import logger
from cnnclassifier.utils.common import get_size
from cnnclassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self)-> str:
       '''Fetch data from url'''

       try:
           dataset_url = self.config.source_URL
           zip_download_path = self.config.local_data_file
           os.makedirs("artifacts/data_ingestion", exist_ok=True)
           logger.info(f"Downloading data from :[{dataset_url}] into :[{zip_download_path}]")

           if "drive.google.com" in dataset_url and "/file/d/" in dataset_url:
               import re
               match = re.search(r"/file/d/([^/]+)", dataset_url)
               if match:
                   file_id = match.group(1)
                   dataset_url = f"https://drive.google.com/uc?export=download&id={file_id}"

           gdown.download(dataset_url, str(zip_download_path), quiet=False)

           logger.info(f"Downloaded data from :[{dataset_url}] into :[{zip_download_path}] of size :[{get_size(zip_download_path)}]")

       except Exception as e:
             raise e


    def extract_zip_file(self):
        """Extract zip file to specified directory"""
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            logger.info(f"Extracting zip file :[{self.config.local_data_file}] into dir :[{unzip_path}]")
            zip_ref.extractall(unzip_path)
            logger.info(f"Extracted zip file :[{self.config.local_data_file}] into dir :[{unzip_path}]")