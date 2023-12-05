import pandas as pd
from hmmlearn import hmm
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

class DataPreprocessing:

    def __init__(self, fileName):
        self.fileName = fileName

    def create_df(self):
        df = pd.read_csv(self.fileName, delimiter=';', encoding='ISO-8859-1')

        return df

    def rename_columns(self, dataset):
        dataset = dataset.rename(columns={"Einsteiger": "Boardings", "Aussteiger": "Alightings", "Zugnr": "TrainID","dtmIstAnkunftDatum": "Arrival", "dtmIstAbfahrtDatum": "Departure", "DS100 kurz": "StationNameShort", "strKurzbezeichnung": "sBahnID"})

        return dataset
    
    def change_datatypes(self, dataset):
        def add_seconds_if_missing(timestamp):
            if len(timestamp.split(':')) == 2:  # If there are only hours and minutes
                return timestamp + ':00'  # Add ':00' for seconds
            return timestamp  
        
        dataset["Departure"] = dataset["Departure"].apply(add_seconds_if_missing)
        dataset["Arrival"] = dataset["Arrival"].apply(add_seconds_if_missing)
        dataset["Arrival"] = pd.to_datetime(dataset["Arrival"], format='%d.%m.%Y %H:%M:%S')
        dataset["Departure"] = pd.to_datetime(dataset["Departure"], format='%d.%m.%Y %H:%M:%S')
        dataset["Boardings"] = dataset['Boardings'].str.replace(',', '.').astype(float).astype(int)
        dataset["Alightings"] = dataset['Alightings'].str.replace(',', '.').astype(float).astype(int)
        dataset['ArrivalDate'] = dataset['Arrival'].dt.date

        return dataset
    
    def hidden_states(self, df):
        stations = df.loc[df['sBahnID'] == 'S1']['Station'].unique()
        #df['StationSequence']  = df.groupby('TrainID')['Station'].transform(lambda x: ','.join(x))
        #stations =  df[['StationSequence']].drop_duplicates().reset_index(drop=True).apply(list)
        #f = df[['StationSequence']].drop_duplicates().reset_index(drop=True)
        #print(stations)
        #enumerated_journeys = {index: journey for index, journey in enumerate(f)}

        #print(enumerated_journeys)
        
        return len(stations)
    
    def observations(self, df):
        sequences = df.groupby('TrainID').apply(lambda x: x[['Boardings', 'Alightings']].values.tolist())
        sequences = np.array(sequences)
        observations = [obs for sequence in sequences for obs in sequence]
        lengths = [len(sequence) for sequence in sequences]
        # Map station names to indices
        station_to_index = {station: idx for idx, station in enumerate(df['Station'].unique())}
        index_to_station = {idx: station for station, idx in station_to_index.items()}

        # Prepare observations for HMM
        #observations = df[['Boardings', 'Alightings']].values.tolist()
        #sequences = np.array(observations)
       #observations = [obs for sequence in sequences for obs in sequence]
        #lengths = [len(sequence) for sequence in sequences]


        return observations, lengths, index_to_station
    
    def going_to_next_station(self,df):
        grouped = df.groupby(['TrainID', 'ArrivalDate'])
        df['on_train'] = grouped['Boardings'].transform(lambda x: x) - grouped['Alightings'].transform(lambda x: x)
        df['going_next_station'] = grouped['on_train'].cumsum()
        '''for i in df["TrainID"].unique():
            for date in df.loc[(df['TrainID'] == i)]['ArrivalDate'].unique():
                df.loc[(df['TrainID'] == i) & (df['ArrivalDate'] == date), 'on_train'] = df.loc[(df['TrainID'] == i) & (df['ArrivalDate'] == date)]['Boardings'] - df.loc[(df['TrainID'] == i) & (df['ArrivalDate'] == date)]['Alightings']
                df.loc[(df['TrainID'] == i) & (df['ArrivalDate'] == date), 'going_next_station'] = df.loc[(df['TrainID'] == i)  & (df['ArrivalDate'] == date)]['on_train'].cumsum()
        '''
        return df
    

    def calculate_emission_probabilities(self, df):
        stations = df['Station'].unique()
        n_stations = len(stations)
        emission_probabilities = np.zeros((n_stations, 2))

        total_boardings = df['Boardings'].sum()
        total_alightings = df['Alightings'].sum()

        for i, station in enumerate(stations):
            # Calculate boarding probability
            #total_boardings = df[df['Station'] == station]['Boardings'].sum()
            avg_boardings = df[df['Station'] == station]['Boardings'].mean() / total_boardings if total_boardings > 0 else 0

            # Calculate alighting probability
            #total_alightings = df[df['Station'] == station]['Alightings'].sum()
            avg_alightings = df[df['Station'] == station]['Alightings'].sum() / total_alightings if total_alightings > 0 else 0

            # Populate the emission probabilities
            emission_probabilities[i, 0] = avg_boardings
            emission_probabilities[i, 1] = avg_alightings

        return emission_probabilities
    
    def calculate_initial_state(self, df):

        # Calculate the total number of boardings
        total_boardings = df['Boardings'].sum()
        station_boardings = df.groupby('Station')['Boardings'].sum().reset_index()


        # Calculate the probability of starting from each station
        station_boardings['Start_Prob'] = station_boardings['Boardings'] / total_boardings

        # Create the initial state distribution
        initial_state_distribution = station_boardings['Start_Prob'].values

        return np.array(initial_state_distribution)


    def calculate_transition_matrix(self, df):

        df = self.going_to_next_station(df)

        aggregated_data = df.groupby('Station').sum(['Boardings', 'going_next_station']).reset_index()

        # Calculate the continuing passengers and transition probabilities
        #aggregated_data['Continuing'] = aggregated_data['Boardings'] - aggregated_data['Alightings'].shift(-1).fillna(0)
        aggregated_data['Transition_Prob'] =  aggregated_data['Boardings'] / aggregated_data['going_next_station'] 
        aggregated_data['Transition_Prob'] = aggregated_data['Transition_Prob'].fillna(0).clip(upper=1)
        # Create the transition matrix for the train line
        n_stations = len(aggregated_data)
        transition_matrix = np.zeros((n_stations, n_stations))

        for i in range(n_stations - 1):
            transition_matrix[i, i + 1] = aggregated_data.iloc[i]['Transition_Prob']

        # The last station has no further stations, so we set its probability to stay there as 1
        transition_matrix[-1, -1] = 1

        normalized_transition_matrix = np.zeros_like(transition_matrix)

        for i in range(len(transition_matrix)):
            row_sum = np.sum(transition_matrix[i])
            if row_sum > 0:
                normalized_transition_matrix[i] = transition_matrix[i] / row_sum
            else:
                # If the sum of the row is 0, this might indicate a state from which no transitions occur
                # In such cases, you can handle it based on your specific scenario
                # For example, set the transition to itself as 1
                normalized_transition_matrix[i, i] = 1

        return normalized_transition_matrix




            


            


