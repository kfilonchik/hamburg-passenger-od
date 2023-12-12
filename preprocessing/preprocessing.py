import pandas as pd
from hmmlearn import hmm
import numpy as np
import numpy as np
from sklearn.preprocessing import Normalizer

class DataPreprocessing:

    def __init__(self, fileName):
        self.fileName = fileName

    def create_df(self):
        df = pd.read_csv(self.fileName, delimiter=';', encoding='ISO-8859-1')

        return df
    
    def create_formatted(self):
        df = pd.read_csv(self.fileName, delimiter=';', encoding='utf-8')

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
        dataset['ArrivalHour'] = dataset['Arrival'].dt.hour

        return dataset
    
    def add_trip_id(self, df):
        # Initialize a column for TripID
        df['TripID'] = 0        

        # Variable to hold the current TripID
        current_trip_id = 1
        # Iterate over the DataFrame
        for i in range(1, len(df)):
            # Check if the current row is part of the same trip as the previous row
    
            if df.iloc[i]['TrainID'] == df.iloc[i-1]['TrainID']:
                # If it is the same train, assign the same TripID as the previous row

                df.iloc[i, df.columns.get_loc('TripID')] = current_trip_id

            else:
             # If it's a different train, increment the TripID and assign it to the current row
                current_trip_id += 1
                df.iloc[i, df.columns.get_loc('TripID')] = current_trip_id
        # Assign the first row a TripID (as the loop starts from the second row)

        df.iloc[0, df.columns.get_loc('TripID')] = 1
    
        return df


    def hidden_states(self, df):
        stations = df['Station'].unique()

        return len(stations)
    
    def observations(self, df):
        stations = len(df['Station'].unique())
        sequences = df.groupby('TripID').apply(lambda x: x[['Boardings'#, 'Alightings'
                                                            ]].values.tolist())
        
        #sequences = np.array(sequences)
        #print(sequences)
        observations = [obs for sequence in sequences for obs in sequence]
        # Map station names to indices
        station_to_index = {station: idx for idx, station in enumerate(df['Station'].unique())}
        index_to_station = {idx: station for station, idx in station_to_index.items()}
        #max_length = max(len(seq) for seq in sequences)
        #print(max_length)
        #observations = [seq + [[0, 0]] * (max_length - len(seq)) for seq in sequences]
        observations =  np.array([np.array(obs).flatten() for obs in observations])
  
        #print([i for i,x in enumerate(observations) if len(x) != 30])
        #observations = sequences
        lengths = [len(sequence) for sequence in observations]
        observations = np.array(observations)
        print(observations[:10])
        print(observations.shape)

        #nsamples, nx, ny = observations.shape
        #observations = observations.reshape((nsamples,nx*ny))

        #observations = np.array(observations)
        #observations =  np.reshape(observations,[len(observations),stations])
        #print(observations)

        return observations, lengths, index_to_station
    
    def going_to_next_station(self,df):
        grouped = df.groupby(['TripID'])
        df['on_train'] = grouped['Boardings'].transform(lambda x: x) - grouped['Alightings'].transform(lambda x: x)
        df['going_next_station'] = grouped['on_train'].cumsum()
      
        return df
    
    
    def impute_missing_boardings(self, group):
        for idx in range(0, len(group)-1):
            diff = group.iloc[idx]['going_next_station'] - group.iloc[idx + 1]['Alightings']
               
            if diff < 0:
                group.iloc[idx, group.columns.get_loc('Boardings')]  += abs(diff)
                group['on_train'] = group['Boardings'].transform(lambda x: x) - group['Alightings'].transform(lambda x: x)
                group['going_next_station'] = group['on_train'].cumsum()

            if (idx == (len(group) - 2)) and (group.iloc[idx]['going_next_station'] - group.iloc[idx+1]['Alightings'] > 0):
                diff = group.iloc[idx]['going_next_station'] - group.iloc[idx+1]['Alightings']
                group.iloc[idx+1, group.columns.get_loc('Alightings')]  += abs(diff)
                group['on_train'] = group['Boardings'].transform(lambda x: x) - group['Alightings'].transform(lambda x: x)
                group['going_next_station'] = group['on_train'].cumsum()

        #group['TripID'] = group['TrainID'].unique()[0] + '_' + str(randint(0, 600000))
        if group['Boardings'].sum() != group['Alightings'].sum():
            print(group['TrainID'].unique(), group['Boardings'].sum(), group['Alightings'].sum())
            print(group)
        return group
        

    def calculate_emission_probabilities(self, df):
        '''
        stations = df['Station'].unique()
        n_stations = len(stations)

        for i, station in enumerate(df['Station'].unique()):
            station_data = df[df['Station'] == station]
            
            # Calculate frequency distribution of boarding categories
            boarding_counts = station_data['Boardings'].sum()
            total_boardings = df['Boardings'].sum()

            # Fill in emission probabilities for the station
            for j, station in enumerate(df['Station'].unique()):
                emission_matrix[i, j] = boarding_counts.get(category, 0) / total_boardings

        # Normalize emission matrix
        emission_matrix /= np.sum(emission_matrix, axis=1, keepdims=True)

        # Handle any NaN values due to division by zero
        emission_matrix = np.nan_to_num(emission_matrix)
        '''
        stations = df['Station'].unique()
        n_stations = len(stations)
        emission_probabilities = np.zeros((n_stations, 2))

        smoothing_constant = 1e-8

        total_boardings = df['Boardings'].sum() + smoothing_constant * len(df['Station'].unique())
        total_alightings = df['Alightings'].sum()



        for i, station in enumerate(stations):
           
            avg_boardings = df[df['Station'] == station]['Boardings'].mean() + smoothing_constant / total_boardings if total_boardings > 0 else 0
            avg_alightings = df[df['Station'] == station]['Alightings'].sum() + smoothing_constant / total_alightings if total_alightings > 0 else 0

            # Populate the emission probabilities
            emission_probabilities[i, 0] = avg_boardings
            emission_probabilities[i, 1] = avg_alightings

        # Check for NaN values
        if np.isnan(emission_probabilities).any():
            mission_probabilities = np.nan_to_num(mission_probabilities)
        
        return emission_probabilities
    
    def calculate_initial_state(self, df):

        # Calculate the total number of boardings
        total_alightings= df['Alightings'].sum()
        station_alightings = df.groupby('Station')['Alightings'].sum().reset_index()

        # Adding a small constant for smoothing
        smoothing_constant = 1e-8

        # Calculate the total number of boardings
        total_alightings = df['Alightings'].sum() + smoothing_constant * len(df['Station'].unique())
        station_alightings = df.groupby('Station')['Alightings'].sum().reset_index()

        # Calculate the probability of starting from each station
        station_alightings['Start_Prob'] = (station_alightings['Alightings'] + smoothing_constant) / total_alightings

        # Normalize to ensure it sums to 1
        normalized_startprob = station_alightings['Start_Prob'].values
        normalized_startprob /= np.sum(normalized_startprob)
        
        return np.array(normalized_startprob)


    def calculate_transition_matrix(self, df):
        '''
        num_stations = len(df['Station'].unique()) # Total number of stations
        transition_matrix = np.zeros((num_stations, num_stations))

        for i, station in enumerate(df['Station'].unique()):
            station_data = df[df['Station'] == station]
            
            # Calculate frequency distribution of boarding categories
            boardings = station_data['going_next_station'].sum()
            alightings = station_data.loc[station_data['Station']== i+1]['Alightings'].sum()
            print(boardings, alightings)

            if boardings > 0:
                 transition_matrix[i, i+1] = alightings / boardings

        # Normalize the transition matrix
        transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

        # Handle any NaN values due to division by zero
        transition_matrix = np.nan_to_num(transition_matrix)

        '''

        df = self.going_to_next_station(df)

        aggregated_data = df.groupby('Station').sum(['Boardings', 'going_next_station']).reset_index()

        # Calculate the continuing passengers and transition probabilities
        #aggregated_data['Continuing'] = aggregated_data['Boardings'] - aggregated_data['Alightings'].shift(-1).fillna(0)
        aggregated_data['Transition_Prob'] =  aggregated_data['Boardings'] / aggregated_data['going_next_station'] 
        aggregated_data['Transition_Prob'] = aggregated_data['Transition_Prob'].fillna(0).clip(upper=1)
        # Create the transition matrix for the train line
        n_stations = len(aggregated_data)
        transition_matrix = np.zeros((n_stations, n_stations))

        smoothing_constant = 1e-8

        for i in range(n_stations - 1):
            transition_matrix[i, i + 1] = aggregated_data.iloc[i]['Transition_Prob']  + smoothing_constant

        # The last station has no further stations, so we set its probability to stay there as 1
        transition_matrix[-1, -1] = 1

        #normalized_transition_matrix = np.zeros_like(transition_matrix)
        #transition_matrix[-1, -1] = 1

        # Normalize the transition matrix
        for i in range(n_stations):
            row_sum = np.sum(transition_matrix[i])
            if row_sum > 0:
                transition_matrix[i] /= row_sum
            else:
                # Handle the case where no transitions occur
                transition_matrix[i, i] = 1
      
        '''
        for i in range(len(transition_matrix)):
            row_sum = np.sum(transition_matrix[i]) 
            if row_sum > 0:
                normalized_transition_matrix[i] = transition_matrix[i] / row_sum
            else:
                # If the sum of the row is 0, this might indicate a state from which no transitions occur
                # In such cases, you can handle it based on your specific scenario
                # For example, set the transition to itself as 1
                normalized_transition_matrix[i, i] = 1
        '''

        # Check for NaN values
        #if np.isnan(transition_matrix).any():
        #    transition_matrix = np.nan_to_num(transition_matrix)
    
        return transition_matrix




            


            


