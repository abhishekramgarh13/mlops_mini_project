import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml
import logging

#logging configure
logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

def load_params(file_name:str) -> float:
    try:
        test_size = yaml.safe_load(open(file_name,'r'))['data_ingestion']['test_size']
        logger.info("parameter loaded successfully")
        return test_size    
    except Exception as e:
        logger.error(f"Error while loading parameter {e}")
        raise e

    

# ingesting data from URL
def load_data(url: str) -> pd.DataFrame:
    try : 
        df = pd.read_csv(url)
        logger.info("data load is successful")
        return df
    except Exception as e:
        logger.error(f"error in load data {e}")
        raise e


def process_data(df:pd.DataFrame)->pd.DataFrame:
    try:
        #removing tweet_id column
        df.drop(columns=['tweet_id'],inplace=True)
        final_df = df[df['sentiment'].isin(['happiness','sadness'])]

        final_df['sentiment'].replace({'happiness':1, 'sadness':0},inplace=True)
        
        logger.info("process data has complited")
        return final_df
    except Exception as e:
        logger.error(f"error in process data {e}")
        raise e



def save_data(data_path:str, train_data:pd.DataFrame, test_data:pd.DataFrame)-> None:
    try :
        os.makedirs(data_path)
        train_data.to_csv(os.path.join(data_path,"train.csv"))
        test_data.to_csv(os.path.join(data_path,'test.csv'))
        logger.info("data save successfully in raw file")
    except Exception as e:
        logger.error(f"error in save data {e}")
        raise e


 

def main():
    test_size=load_params('params.yaml')
    df = load_data("https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv")
    final_df = process_data(df)
    train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
    data_path = os.path.join('data','raw')
    save_data(data_path,train_data,test_data)

if __name__ == "__main__":
    main()

