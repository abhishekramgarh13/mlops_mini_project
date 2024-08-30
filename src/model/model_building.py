import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.linear_model import LogisticRegression
import yaml


logger = logging.getLogger("Model training")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

#fetch data from data/features
def load_features_data(file_path):
    try:
        train_data = pd.read_csv(file_path)

        X_train = train_data.iloc[:,0:-1].values
        y_train = train_data.iloc[:,-1].values
        logger.info("featched data from data/features succesfully.")
        return X_train,y_train
    except Exception as  e:
        logger.error(f"error in featching data {e}.")
        raise e

def train_model(X_train,y_train):
    try:
        
        logger.info("paramerter loaded successfully.")
        # define and training the GradientBoostingClassifier
        clf = LogisticRegression(C=1,solver='liblinear',penalty='l2')
        clf.fit(X_train, y_train)
        logger.info("model trained successfully.")

        #saving model
        with open('./models/model.pkl', 'wb') as file:
            pickle.dump(clf, file)

        logger.info("model saved successfully.")

    except Exception as e:
        logger.error(f"error while model training {e}.")
        raise e
    
def main():
    X_train,y_train = load_features_data("./data/features/train_bow.csv")
    train_model(X_train,y_train)

if __name__ == "__main__":
    main()