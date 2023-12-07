from preprocessing.preprocessing import DataPreprocessing
from training.train import DataTraining
from mining.routes import RoutesMining
from hmmlearn import hmm
import pandas as pd
from joblib import dump, load
import numpy as np
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from generate.generate import GenerateData

def start_preprocessing(preprocessing):
    df = preprocessing.create_df()
    df = preprocessing.rename_columns(df)
    df = preprocessing.change_datatypes(df)

    df = df.loc[df['sBahnID'] == 'S1'] #only s-bahn line

    return df

def add_smoothing_to_transmat(transmat, smoothing_constant=1e-8):
    smoothed_transmat = transmat + smoothing_constant
    normalized_transmat = np.zeros_like(smoothed_transmat)

    for i, row in enumerate(smoothed_transmat):
        normalized_transmat[i] = row / np.sum(row)
    
    return normalized_transmat


def create_train_test_valid(df):
    unique_train_ids = df['TrainID'].unique()

    # Shuffle and split the Train IDs
    train_ids, test_ids = train_test_split(unique_train_ids, test_size=0.2, random_state=42)
    test_ids, val_ids = train_test_split(test_ids, test_size=0.5, random_state=42)

    # Reconstruct the datasets based on Train IDs
    train_df = df[df['TrainID'].isin(train_ids)]
    test_df = df[df['TrainID'].isin(test_ids)]
    val_df = df[df['TrainID'].isin(val_ids)]

    return train_df, test_df, val_df

def evaluate_model(preprocessing, df, observations):
    y_test = list(df['Station'])
    log_likelihood = model.score(observations)

    # Predict the state sequence for the test set
    predicted_states = model.predict(observations)

    # Calculate accuracy if true states (y_test) are known
    accuracy = accuracy_score(y_test, predicted_states)

    return log_likelihood, accuracy


if __name__== '__main__':
    # Start pre-processing
    preprocessing = DataPreprocessing("data/sbahn_hamburg.csv")

    df = start_preprocessing(preprocessing)

    # Split datasets into train, test and validation
    train, test, validation = create_train_test_valid(df)

    observations_train, lengths_train, index_to_station_train = preprocessing.observations(train)
    observations_test, lengths_test, index_to_station_test = preprocessing.observations(test)
    observations_valid, lengths_valid, index_to_station_valid = preprocessing.observations(validation)

    # Start training
    training = DataTraining(preprocessing, df, observations_train, lengths_train)
    model = training.setup_hmmodel()

    model = load('model/hmm_model.joblib')

    generator = GenerateData(df, index_to_station_test, test, model)
    eventlog = generator.generate_data()



    

