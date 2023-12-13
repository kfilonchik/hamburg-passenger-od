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
    unique_train_ids = df['TripID'].unique()

    # Shuffle and split the Train IDs
    train_ids, test_ids = train_test_split(unique_train_ids, test_size=0.3, random_state=42)
    test_ids, val_ids = train_test_split(test_ids, test_size=0.5, random_state=42)

    # Reconstruct the datasets based on Train IDs
    train_df = df[df['TripID'].isin(train_ids)]
    test_df = df[df['TripID'].isin(test_ids)]
    val_df = df[df['TripID'].isin(val_ids)]

    return train_df, test_df, val_df

def evaluate_model(df, observations, model):
    y_test = list(df['Station'])
    log_likelihood = model.score(observations)

    # Predict the state sequence for the test set
    predicted_states = model.predict(observations)

    # Calculate accuracy if true states (y_test) are known
    accuracy = accuracy_score(y_test, predicted_states)

    return log_likelihood, accuracy

def enhance_data(df, preprocessing):
    df = preprocessing.add_trip_id(df)
    df = preprocessing.going_to_next_station(df)
    df = df.groupby(['TripID']).apply(preprocessing.impute_missing_boardings).reset_index(drop=True)
    df.to_csv('data/train_trips.csv', index=False,  sep=';', encoding='utf-8')

    return df



if __name__== '__main__':
    # Start pre-processing
    preprocessing = DataPreprocessing("data/train_trips.csv")
    #df = start_preprocessing(preprocessing)
    df = preprocessing.create_formatted()

    df1 = df.loc[(df['Station']=='Airport')]['TripID'].values.tolist()
    df = df[df['TripID'].isin(df1)]

    # Split datasets into train, test and validation
    train, test, validation = create_train_test_valid(df)

    observations_train, lengths_train, index_to_station_train = preprocessing.observations(train)
    observations_test, lengths_test, index_to_station_test = preprocessing.observations(test)
    observations_valid, lengths_valid, index_to_station_valid = preprocessing.observations(validation)
    
    trip = df.loc[df['TripID'] == 1]

    training = DataTraining(preprocessing, df, observations_train, lengths_train, trip)
    #model = training.setup_hmmodel()
    #model = training.fit_multinominal_model_parameters()
    #transit = df.loc[df['TripID'] == 1].groupby('TripID').apply(preprocessing.calculate_transition_matrix)
    #print(preprocessing.calculate_means(train))
    #t = preprocessing.calculate_means(train)
    #print(t.shape)
    #model = training.fit_gauss_model_parameters()
 
    
    #print('Initial probabilities:', preprocessing.calculate_initial_state(train.loc[train['TripID'] == 2]))
    #print('Transition probabilities:', preprocessing.calculate_transition_matrix(train.loc[train['TripID'] == 2]))
    #print('Emission probabilities:', preprocessing.calculate_emission_probabilities(train.loc[train['TripID'] == 2]))


    # Start training
    #training = DataTraining(preprocessing, df, observations_train, lengths_train)
    #model = training.setup_hmmodel()

    model = load('hmm_model_trained_9.joblib')
    #score =  model.score(observations_valid[10:20])
    #print(score)
    #print(model.transmat_)

    # Generate samples
    #X, Z = model.sample(10)
    #print(X, Z, index_to_station_valid)

    #print(model.decode(observations_valid[300:310]))

    #l, a = evaluate_model(df, observations_test[0:2], model)
    #print('liklihood', l)
    #print('accuracy', a)

    generator = GenerateData(df, index_to_station_test, validation, observations_valid,  model, preprocessing)
    eventlog = generator.generate_data_2()



    

