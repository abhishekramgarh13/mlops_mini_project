import numpy as np
import pandas as pd
import os
import logging
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle
import yaml

logger = logging.getLogger("Feature Engineering")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

#fetch data from data/processed
def read_processed_data() -> pd.DataFrame:
    try :
        train_data = pd.read_csv('./data/processed/train_processed.csv')
        test_data = pd.read_csv('./data/processed/test_processed.csv')
        logger.info("processed data loaded successfully")
        return train_data,test_data
    except Exception as e:
        logger.error(f"error in loading processed data {e}")
        raise e


def replace_null(train_data,test_data) :
    try:
        train_data.fillna('',inplace=True)
        test_data.fillna('',inplace=True)
        logger.info("null value replaced successfully")
        return train_data,test_data
    except Exception as e:
        logger.error(f"error in replacing null {e}")
        raise e

def param_load(file_path:str)-> float:
    try:
       max_features = yaml.safe_load(open(file_path,'r'))['feature_engineering']['max_features']
       logger.info("parameter loaded successfully")
       return max_features
    except Exception as e:
        logger.error(f"error in loading parameter {e}.")
        raise e


# Apply Tfidf vectorizer
def bagOfWords(X_train, X_test,max_features):
    try:
        
        vectorizer = CountVectorizer(max_features=max_features)

        # Fit the vectorizer on the training data and transform it
        X_train_bow = vectorizer.fit_transform(X_train)

        # Transform the test data using the same vectorizer
        X_test_bow = vectorizer.transform(X_test)
        
        logger.info("succesfully appled bow on X_train and X_test data.")
        return X_train_bow,X_test_bow

    except Exception as e:
        logger.error(f"error in apply bow on data {e}")
        raise e

#store the data inside data/feature
def save_data(train_df,test_df):
    try:
        data_path = os.path.join('data','features') 
        os.makedirs(data_path)

        train_df.to_csv(os.path.join(data_path,"train_bow.csv"))
        test_df.to_csv(os.path.join(data_path,'test_bow.csv'))
        logger.info('bow data save successfully')
    except Exception as e:
        logger.error(f"error in svaing feature engineering data {e}.")
        raise e
    
def main():
    train_data,test_data=read_processed_data()
    train_data,test_data=replace_null(train_data,test_data)
    X_train = train_data['content'].values
    y_train = train_data['sentiment'].values
    X_test = test_data['content'].values
    y_test = test_data['sentiment'].values
    max_features = param_load('params.yaml')
    X_train_bow,X_test_bow = bagOfWords(X_train,X_test,max_features)
    train_df = pd.DataFrame(X_train_bow.toarray())
    train_df['label'] = y_train
    test_df = pd.DataFrame(X_test_bow.toarray())
    test_df['label'] = y_test
    save_data(train_df,test_df)

if __name__=="__main__":
    main()

