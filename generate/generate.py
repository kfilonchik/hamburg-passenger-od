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

    def generate_data_first(self):

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
    

    def generate_data_second(self):
        self.test['station_id'] = 0

        self.test = self.test.groupby(['TripID']).apply(self.preprocessing.numerate_stations)

        stations = self.df['Station'].unique().tolist()
        passengerid = 1

        #self.test = self.test[560:660] 
       # _, sequence_indices = self.model.decode(np.array(self.test['Boardings'].values.tolist()).reshape(-1,1), algorithm='viterbi') # decoding tuples

        evlog = pd.DataFrame()

        size_seq = [10, 2, 4, 3 , 12, 7, 8, 10, 13, 5, 7, 3, 2, 4, 10]
        start = [800, 1000, 2000, 5000, 8000, 9000, 10000, 50, 15000, 12000, 5, 16000, 17400, 17800, 16425]

        #self.test = self.test[500:550] 
        #_, sequence_indices = self.model.decode(np.array(self.test['Boardings'].values.tolist()).reshape(-1,1), algorithm='viterbi') # decoding tuples

        for i in range(len(size_seq)):
            index1 = start[i]
            index2 = start[i] + size_seq[i]

            t = self.test[index1:index2] 
            _, sequence_indices = self.model.decode(np.array(t['Boardings'].values.tolist()).reshape(-1,1), algorithm='viterbi') # decoding tuples

            for i in range(len(t)):
                if sequence_indices[i] > t[i:i+1]['station_id'].values:
                    data = t[i:sequence_indices[i]]
                    data['PassengerID'] = passengerid
                    evlog = pd.concat([evlog, data])

                passengerid = passengerid + 1
            

        evlog.to_csv('data/event_log5.csv', index=False,  sep=';', encoding='utf-8')



        