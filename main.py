from preprocessing.preprocessing import DataPreprocessing
from mining.routes import RoutesMining
from hmmlearn import hmm
import pandas as pd
from joblib import dump, load
import numpy as np
from functools import reduce

def start_preprocessing(preprocessing):
    df = preprocessing.create_df()
    df = preprocessing.rename_columns(df)
    df = preprocessing.change_datatypes(df)

    df = df.loc[df['sBahnID'] == 'S1']


    return df


def Average(lst): 
    return reduce(lambda a, b: a + b, lst) / len(lst) 


if __name__== '__main__':
    preprocessing = DataPreprocessing("data/sbahn_hamburg.csv")
    df = start_preprocessing(preprocessing)
    observations, lenghs, index_to_station = preprocessing.observations(df)

    # Hidden Markov Model, to start take out comments
    '''
    print(preprocessing.calculate_emission_probabilities(df))
    model = hmm.MultinomialHMM(n_components=preprocessing.hidden_states(df), init_params='')
    model.n_features = len(observations[0])
    model.startprob_ = preprocessing.calculate_initial_state(df)
    #print(preprocessing.calculate_initial_state(df))
    model.transmat_ = preprocessing.calculate_transition_matrix(df)
    #print(preprocessing.calculate_transition_matrix(df))
    model.emissionprob_ = preprocessing.calculate_emission_probabilities(df)

    model.fit(observations, lenghs)


    dump(model, 'hmm_model.joblib')
    '''
    model = load('hmm_model.joblib')
    passenger_id = 1
    event_log = []

    obs = []
    passengers = []

    for index, row in df[20:25].iterrows():
        print(row['Station'])

        obs.append([row['Boardings'], row['Alightings']])
        passengers.append(row['Boardings'])


    

       #for _ in range(row['Boardings']):
           # print(df[:2]['Station'])

            ##obsr.append(row['Alightings'])
            #print(np.array(obsr).astype(int))
        
            #print(row['Station'], row['Arrival'], [row['Boardings'], row['Alightings']])
            # Generate the sequence of stations for this passenger
            # In this example, we just use the index for simplicity
            # In practice, you'd use the HMM to predict this sequence
    _, sequence_indices = model.decode(obs, algorithm='viterbi')
    print(sequence_indices)
    print(obs)


            # Create an event log entry for each station in the sequence
    for seq_index in sequence_indices:
        df.iloc[seq_index]['Station']
        df.iloc[seq_index]['Arrival']
        event_log.append({
            'PassengerID': round(Average(passengers)),
            'Station': df.iloc[seq_index]['Station'],
            'Timestamp': df.iloc[seq_index]['Arrival']
        })

        

            #passenger_id += 1


    event_log = pd.DataFrame(event_log)
    event_log.to_csv('data/event_log2.csv', index=False,  sep=';', encoding='utf-8')
    '''
    individual_routes = []
    passenger_id = 0
    def process_journey(group):
        global passenger_id
        observations = group[['Boardings', 'Alightings']].to_numpy()
        _, station_sequence = model.decode(observations, algorithm="viterbi")

        for _, row in group.iterrows():
            for _ in range(row['Boardings']):
                individual_route = {
                    'PassengerID': passenger_id,
                    'StationSequence': station_sequence,
                    'Timestamp': row['Arrival']
                }
                individual_routes.append(individual_route)
                passenger_id += 1
'''
    # Apply the HMM to find the most likely sequence of stations
   # logprob, stations = model.decode(observations, algorithm="viterbi")

    # Generate individual routes
  #  individual_routes = []
  #  passenger_id = 1



    #print(df)
    # Apply the function to each group
    #df.groupby('TrainID').apply(process_journey)

    # Convert the individual routes to a DataFrame
    #individual_routes_df = pd.DataFrame(individual_routes)
    #individual_routes_df.to_csv('data/event_log.csv', index=False,  sep=';', encoding='utf-8')

    # Predict the most probable sequence of hidden states for the given observation sequence
    #hidden_states = model.predict(observations)


    #print(df.groupby('TrainID')['Station'].transform(lambda x: ','.join(x)))
    #df['StationSequence']  = df.groupby('TrainID')['Station'].transform(lambda x: ','.join(x))
    #k =  df[['StationSequence']].drop_duplicates().reset_index(drop=True)
    #k.to_csv('data/routes.csv', index=False,  sep=';', encoding='utf-8')

    #mining = RoutesMining(df)
    #routes = mining.route_mining_simple(df)
    #routes= mining.read_eventlog("data/passenger_paths_new3.csv")
    #eventlog = mining.create_eventlog(df, routes)
    

    



