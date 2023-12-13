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
        #transformer = Normalizer().fit(observations)
        #observations = transformer.transform(observations)

        #print(observations[:10])
        #print(observations.shape)

        #nsamples, nx, ny = observations.shape
        #observations = observations.reshape((nsamples,nx*ny))

        #observations = np.array(observations)
        #observations =  np.reshape(observations,[len(observations),stations])
        #print(observations)

        return observations, lengths, index_to_station
    
    def numerate_stations(self, group):
        i = 0
        for idx in range(0, len(group)-1):
            i = i + 1
            group.iloc[idx, group.columns.get_loc('station_id')] = i 

        return group
    
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
        #emission_probabilities = np.zeros((n_stations, 2))
        emission_probabilities = np.zeros((n_stations, 1))

        smoothing_constant = 1e-8

        total_boardings = df['Boardings'].sum() + smoothing_constant * len(df['Station'].unique())
        #total_alightings = df['Alightings'].sum()



        for i, station in enumerate(stations):
           
            avg_boardings = df[df['Station'] == station]['Boardings'].mean() + smoothing_constant / total_boardings if total_boardings > 0 else 0
            #avg_alightings = df[df['Station'] == station]['Alightings'].sum() + smoothing_constant / total_alightings if total_alightings > 0 else 0

            # Populate the emission probabilities
            emission_probabilities[i, 0] = avg_boardings
            #emission_probabilities[i, 1] = avg_alightings

        # Check for NaN values
        if np.isnan(emission_probabilities).any():
            emission_probabilities = np.nan_to_num(emission_probabilities)
        
        return emission_probabilities
    
    def calculate_initial_state(self, df):

        # Calculate the total number of boardings
        total_alightings= df['Alightings'].sum()
        station_alightings = df.groupby('Station')['Alightings'].sum().reset_index()

        # Adding a small constant for smoothing
        smoothing_constant = 1e-8

        # Calculate the total number of boardings
        total_alightings = df['Alightings'].sum() #+ smoothing_constant * len(df['Station'].unique())
        station_alightings = df.groupby('Station')['Alightings'].sum().reset_index()

        # Calculate the probability of starting from each station
        station_alightings['Start_Prob'] = (df['Alightings'] / df['going_next_station'].shift(1).fillna(0)).fillna(0)#(station_alightings['Alightings'] ) / total_alightings

        start_prob = np.array(station_alightings['Start_Prob'])

        # Normalize to ensure it sums to 1
        normalized_startprob = station_alightings['Start_Prob'].values
        normalized_startprob /= np.sum(normalized_startprob)
        #print(start_prob)

        '''

        normalized_startprob = np.zeros_like(start_prob)
        for i in range(len(start_prob)):
            row_sum = np.sum(start_prob[i]) 
            if row_sum > 0:
                normalized_startprob[i] = start_prob[i] / row_sum
            else:
                # If the sum of the row is 0, this might indicate a state from which no transitions occur
                # In such cases, you can handle it based on your specific scenario
                # For example, set the transition to itself as 1
                normalized_startprob[i] = 0

        # Handle any NaN values due to division by zero
        if np.isnan(start_prob).any():
            start_prob = np.nan_to_num(start_prob)

        '''

        print(normalized_startprob)

        
        return normalized_startprob


    def calculate_transition_matrix(self, fahrt):

        fahrt['percent_leaving'] = (fahrt['Alightings'] / fahrt['going_next_station'].shift(1).fillna(0)).fillna(0)

        for idx in range(0, len(fahrt)-1):  # 1|len(fahrt)-1
            st = str(idx + 1)
            #ps = percentage staying
            fahrt.loc[0:idx, 'ps_' + st] = 1 #set preceding stations to 100%
            j = idx + 1
            for j in range(j, len(fahrt)):
                fahrt.loc[j, 'ps_' + st] = fahrt.loc[j - 1, 'ps_' + st] * (1 - fahrt.loc[j, 'percent_leaving']) # ps = ps(1) * ( 1 - percentatge leaving )

            # # Handle NaN values if necessary (e.g., fill with 0 or drop)
            fahrt.fillna(0)
            
            #pl = propability leaving
            fahrt['pl_' + st ] =  (fahrt['ps_' + st].shift(1) - fahrt['ps_' + st]).fillna(0)

        fahrt['pl_26'] = 1.0
                    
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
        '''
        df = self.going_to_next_station(df)

        aggregated_data = df.groupby('Station').sum(['Alightings', 'going_next_station']).reset_index()

        # Calculate the continuing passengers and transition probabilities
        #aggregated_data['Continuing'] = aggregated_data['Boardings'] - aggregated_data['Alightings'].shift(-1).fillna(0)
        aggregated_data['Transition_Prob'] =  aggregated_data['Alightings'] / aggregated_data['going_next_station'] 
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
    
        return fahrt
    
    def get_pl_columns(self, trip):
        dfpl = pd.DataFrame()
        for x in range(1, len(trip.columns) + 1):  # Assuming you want to iterate based on the number of columns
            column_name = 'pl_' + str(x)
            if column_name in trip.columns:  # Check if column exists
                dfpl[column_name] = trip[column_name]  # Add column to dfpl

        transition_matrix = np.matrix(dfpl)
        print(dfpl)
        print(transition_matrix)
        print(transition_matrix.shape)

        # Normalize the transition matrix
        normalized_transition_matrix = np.zeros_like(transition_matrix)
        for i in range(len(transition_matrix)):
            row_sum = np.sum(transition_matrix[i]) 
            if row_sum > 0:
                normalized_transition_matrix[i] = transition_matrix[i] / row_sum
            else:
                # If the sum of the row is 0, this might indicate a state from which no transitions occur
                # In such cases, you can handle it based on your specific scenario
                # For example, set the transition to itself as 1
                normalized_transition_matrix[i, i] = 0

        # Handle any NaN values due to division by zero
        if np.isnan( normalized_transition_matrix).any():
            normalized_transition_matrix = np.nan_to_num( normalized_transition_matrix)


        return normalized_transition_matrix
        

    def calculate_cov_matrix(self, df):
        boardings = np.array(df['Boardings'].values.tolist())  # Boardings data
        alightings = np.array(df['Alightings'].values.tolist())  # Alightings data

        # Assuming you have a way to segment your data for each state
        # Here, I'm just dividing the data equally as an example
        segment_length = len(df['Station'].unique())
        covariance_matrices = []

        for i in range(len(df['Station'].unique())):
            start_index = i * segment_length
            end_index = start_index + segment_length

            # Segment data
            segment_boardings = boardings[start_index:end_index]
            segment_alightings = alightings[start_index:end_index]

            # Calculate covariance matrix for this segment
            cov_matrix = np.cov(segment_alightings)
            covariance_matrices.append(cov_matrix)
        
        cov_matrix = np.cov(alightings)

        return cov_matrix
    

    def calculate_means(self, df):
        boardings = np.array(df['Boardings'].values.tolist())  # Boardings data
        alightings = np.array(df['Alightings'].values.tolist())  # Alightings data

        # Assuming you have a method to segment your data for each state
        # Here, I'm just dividing the data equally as an example
        segment_length = len(df['Station'].unique()) # num_states is the number of hidden states
        means = []

        for i in range(len(df['Station'].unique())):
            start_index = i * segment_length
            end_index = start_index + segment_length
            # Segment data
            segment_boardings = boardings[start_index:end_index]
            segment_alightings = alightings[start_index:end_index]

            # Calculate mean for this segment
            mean_vector = [np.mean(segment_alightings)]
            means.append(mean_vector)

        means = [np.mean(alightings)]
            

        return np.array(means)






            


            


