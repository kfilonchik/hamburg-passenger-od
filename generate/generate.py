from hmmlearn import hmm
import pandas as pd
from joblib import dump, load
import numpy as np

class GenerateData():

    def __init__(self, df, index_to_station_test, test, observations, model, preprocessing):
        self.df = df
        self.test = test
        self.observations = observations
        self.model = model
        self.index_to_station_test = index_to_station_test
        self.preprocessing = preprocessing

    def generate_data(self):

        event_log = []
        test = self.test[960:968] # generate data based on first 1000 rows

        for index, row in test.iterrows():
            obs = []
            passengers = []
            obs.append([row['Boardings']])
            passengers.append(row['Boardings'])

            _, sequence_indices = self.model.decode(obs, algorithm='viterbi') # decoding tuples
            print(sequence_indices)
            print(obs)

            alighting_stations = [self.index_to_station_test[state] for state in sequence_indices]

            for station in alighting_stations:
                event_log.append({
                    'PassengerID': row['Boardings'],
                    'BoardingStation': row['Station'],
                    'TripID': row['TripID'],
                    'TrainID': row['TrainID'],
                    'AlightingStation': station,
                    'TimestampStart': row['Arrival'],
                    'TimestampEnd':  self.df.loc[(self.df['TrainID'] == row['TrainID']) & (self.df['Station'] == station)]['Arrival']
                })
            
        event_log = pd.DataFrame(event_log)
        event_log.to_csv('data/event_log2.csv', index=False,  sep=';', encoding='utf-8')

        return event_log
    

    def generate_data_2(self):
        self.test['station_id'] = 0

        self.test = self.test.groupby(['TripID']).apply(self.preprocessing.numerate_stations)

        stations = self.df['Station'].unique().tolist()
        passengerid = 1

        #self.test = self.test[560:660] 
       # _, sequence_indices = self.model.decode(np.array(self.test['Boardings'].values.tolist()).reshape(-1,1), algorithm='viterbi') # decoding tuples

        evlog = pd.DataFrame()


        self.test = self.test[500:550] 
        _, sequence_indices = self.model.decode(np.array(self.test['Boardings'].values.tolist()).reshape(-1,1), algorithm='viterbi') # decoding tuples


        for i in range(len(self.test)):
            if sequence_indices[i] > self.test[i:i+1]['station_id'].values:
                data = self.test[i:sequence_indices[i]]
                data['PassengerID'] = passengerid
                evlog = pd.concat([evlog, data])

            passengerid = passengerid + 1
         

        evlog.to_csv('data/event_log3.csv', index=False,  sep=';', encoding='utf-8')



        