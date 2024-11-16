import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# The Inputs required by the DataIngestionComponent will be given by DataIngestionConfig
# The outputs of the DataIngestionComponent are DataIngestionArtifact

#dataclass helps to define the class variables without needing the `init` method.
# It is very useful when we want to define only variables inside class like here.
# If we want to add other functionalities too then we can use regular class.
@dataclass 
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')
    # Now our data ingestion component knows where to save train.csv,test.csv etc.
  
class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion componenet")
        try:
            df=pd.read_csv('notebook/data/stud.csv') # Here data can be read from any source either local or remote
            logging.info("Read the dataset as Dataframe")
            
            # Making the directory and files using the paths present in the config
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True) # Directory
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True) # Raw data file
            
            logging.info("Train Test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True) #Train data file
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True) #Test data file
            
            logging.info("Data Ingestion completed")
            
            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            ) # Returning For data transformation componenet
        except Exception as e:
            raise CustomException(e,sys)
  


# Testing if the component is working or not
# Artifact will be automatically  created
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
    