from hmmlearn import hmm
import pandas as pd
from joblib import dump, load
import numpy as np

class GenerateData():

    def __init__(self, df, index_to_station_test, test, model):
        self.df = df
        self.test = test
        self.model = model
        self.index_to_station_test = index_to_station_test

    def generate_data(self):

        event_log = []
        test = self.test[:1000] # generate data based on first 1000 rows

        for index, row in test.iterrows():
            obs = []
            passengers = []
            obs.append([row['Boardings'], row['Alightings']])
            passengers.append(row['Boardings'])

            _, sequence_indices = self.model.decode(obs, algorithm='viterbi') # decoding tuples
            print(sequence_indices)
            print(obs)

            alighting_stations = [self.index_to_station_test[state] for state in sequence_indices]

            for station in alighting_stations:
                event_log.append({
                    'PassengerID': row['Boardings'],
                    'BoardingStation': row['Station'],
                    'TrainID': row['TrainID'],
                    'AlightingStation': station,
                    'TimestampStart': row['Arrival'],
                    'TimestampEnd':  self.df.loc[(self.df['TrainID'] == row['TrainID']) & (self.df['Station'] == station)]['Arrival']
                })
            
        event_log = pd.DataFrame(event_log)
        event_log.to_csv('data/event_log.csv', index=False,  sep=';', encoding='utf-8')

        return event_log