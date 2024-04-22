import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

import sys
import codecs

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

class DataPreprocessor:
    def __init__(self, max_features=18000):
        
        self.max_features = max_features
        self.vectorizer = None
        self.scaler = None
        self.pipeline = None

    def read_csv(self, file_path):
        # Load data
        df = pd.read_csv(file_path, sep="\t")  # If the separator is tab
    
        # Drop specified columns
        columns_to_drop = ['id', 'metadata', 'region_main', 'region_sub', 'date_str', 'date_circa']
        df = df.drop(columns=columns_to_drop)
    
        return df

    def preprocess(self, texts, date_min, date_max, region_main_id, region_sub_id):

        # Initialize and fit TfidfVectorizer
        self.vectorizer = self.create_tfidf_vectorizer(texts)
        X_text = self.vectorizer.transform(texts).toarray()
        
        # Convert date_min and date_max to numpy arrays
        X_date_min = np.array(date_min).reshape(-1, 1)
        X_date_max = np.array(date_max).reshape(-1, 1)
        X_region_main_id = np.array(region_main_id).reshape(-1, 1)
        X_region_sub_id = np.array(region_sub_id).reshape(-1, 1)
    

        print("X_date_min:", X_date_min)
        # Concatenate the features 
        X = np.concatenate((X_text, X_date_min, X_date_max, X_region_main_id, X_region_sub_id), axis=1)
        
       
        # Initialize and fit MinMaxScaler
        self.scaler = self.create_minmax_scaler(X)
        # Normalize the features
        X_normalized = self.scaler.transform(X)

        return X_normalized

    def create_tfidf_vectorizer(self, texts):

        vectorizer = TfidfVectorizer(max_features=self.max_features)
        vectorizer.fit(texts)
        return vectorizer

    def create_minmax_scaler(self, X):

        scaler = MinMaxScaler()
        scaler.fit(X)
        return scaler

class NNModel:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = self.create_model()

    def create_model(self):
        pass
        

    def train(self, X_train, y_train, epochs, batch_size):
        pass

    def evaluate(self, X_test, y_test):
        pass

    def predict(self, X):
        pass



def main():

    dp = DataPreprocessor()
    df = dp.read_csv("./Data/iphi2802.csv")
    # Assuming 'texts', 'date_min', and 'date_max' are columns in your dataframe
    texts = df['text'].tolist()
    date_min = df['date_min'].tolist()
    date_max = df['date_max'].tolist()
    region_main_id = df['region_main_id'].tolist()
    region_sub_id = df['region_sub_id'].tolist()

    # print("Region Sun ID:", region_sub_id)
    # print(df.head())
    # print(len(region_sub_id))


    X = dp.preprocess(texts, date_min, date_max, region_main_id, region_sub_id)


    


if __name__ == "__main__":
    main()
