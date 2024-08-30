import numpy as np
import pandas as pd
import os
import logging
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('wordnet')
nltk.download('stopwords')

logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

#fetch the data from data/raw
def read_raw_data() -> pd.DataFrame:
    try :
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.info("raw data loaded successfully")
        return train_data,test_data
    except Exception as e:
        logger.error(f"error in loading raw data {e}")
        raise e


#Transform the data

def lemmatization(text):
    try :
        lemmatizer= WordNetLemmatizer()

        text = text.split()
        text=[lemmatizer.lemmatize(y) for y in text]

        t = " " .join(text)
        return t
    
    except Exception as e:
        logger.error(f"error in lemmatization {e}")



def remove_stop_words(text):
    try:
        # Loading stop words
        stop_words = set(stopwords.words("english"))
        
        # Removing stop words from the input text
        filtered_words = [i for i in str(text).split() if i not in stop_words]
        result = " ".join(filtered_words)
        
        return result
    except Exception as e:
        logging.error(f"error in stopwords removeing: {e}.")
        raise e

def removing_numbers(text):
    try:
        text=''.join([i for i in text if not i.isdigit()])
        
        return text
    except Exception as e:
        logger.error(f"error in removing number {e}.")
        raise e

def lower_case(text):
    try:
        text = text.split()

        text=[y.lower() for y in text]
        
        lower = " " .join(text)
       
        return lower
    except Exception as e:
        logger.error(f"error in lower_casing text {e}.")
        raise e

def removing_punctuations(text):
    ## Remove punctuations
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛',"", )

        ## remove extra whitespace
        text = re.sub('\s+', ' ', text)
        text =  " ".join(text.split())
        return text.strip()
    except Exception as e:
        logger.error(f"error in removing punctuations {e}")
        raise e

def removing_urls(text):
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        
        return url_pattern.sub(r'', text)
    except Exception as e:
        logger.error(f"error in removing urls {e}")
        raise e

def remove_small_sentences(df):
    try :
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan
    except Exception as e:
        logger.error(f"error in small sentences {e}.")


def normalize_text(df : pd.DataFrame)-> pd.DataFrame:
    try:
        df.content=df.content.apply(lambda content : lower_case(content))
        logger.info("text xonverted into lowercase successfully.")
        df.content=df.content.apply(lambda content : remove_stop_words(content))
        logging.info("Stop words removed successfully.")
        df.content=df.content.apply(lambda content : removing_numbers(content))
        logger.info("number removerd from text successfully.")
        df.content=df.content.apply(lambda content : removing_punctuations(content))
        df.content=df.content.apply(lambda content : removing_urls(content))
        logger.info("url has removed successfully from text.")
        df.content=df.content.apply(lambda content : lemmatization(content))
        logger.info("lemmatization run successfully")
        return df
    except Exception as e:
        logger.error(f"error in normalizing text {e}.")
        raise e



#store the preprocessed data
def save_processed_data(train_processed_data, test_processed_data):
    try:
        data_path = os.path.join('data','processed') 
        os.makedirs(data_path)

        train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"))
        test_processed_data.to_csv(os.path.join(data_path,'test_processed.csv'))
        logger.info("Processed data saved successfully")
    except Exception as e:
        logger.error(f"error in saving processed data {e}.")
        raise e
    
def main():
    train_data, test_data = read_raw_data()
    train_processed_data = normalize_text(train_data)
    test_processed_data = normalize_text(test_data)
    save_processed_data(train_processed_data,test_processed_data)

if __name__=="__main__":
    main()