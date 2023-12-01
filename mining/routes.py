import pandas as pd
import random

class RoutesMining:

    def __init__(self, df):
        #Dataframe
        self.df = df

    def read_eventlog(self, fileName):
        df = pd.read_csv(fileName, delimiter=';', encoding='utf-8')

        return df
    
    def get_next_stations(self, station_order_df, current_station, train_id):
        # This function returns the list of next stations for a given current_station within the same TrainID
        stations = station_order_df[station_order_df['TrainID'] == train_id]
        current_index = stations[stations['Station'] == current_station].index[0]
        next_stations = stations.loc[current_index + 1:]
        return next_stations['Station'].tolist()

    def route_mining_simple(self, df):
         #Initialize an empty list to store passenger paths
        passenger_paths = []
        # Counter for unique Passenger IDs
        passenger_id = 1
        # A dictionary to keep track of passengers currently on the train
        # Key: PassengerID, Value: (boarding station, boarding time)
        passengers_on_train = {}
        # Assuming 'train_data' is the DataFrame for the specific train journey
        # It should contain 'TrainID', 'Timestamp', 'Station', 'Einsteiger' (boardings), and 'Aussteiger' (alightings)

         # Iterate over each unique TrainID
        for train_id, train_group in df.groupby('TrainID'):
            passengers_on_train = {}

            # Iterate over each row in the train data
            for index, row in train_group.iterrows():
                station = row['Station']
                timestamp = row['Arrival']
                train_id = row['TrainID']
                sbahn_id = row ['sBahnID']
                boardings = row['Boardings']
                alightings = row['Alightings']


                # Assign Passenger IDs to each boarding passenger
                # Board new passengers
                if boardings > 0:
                    #print('this is boarding', boardings)
                    for _ in range(boardings):
                        passengers_on_train[passenger_id] = {'BoardingStation': station, 'BoardingTime': timestamp}
                        passenger_id += 1

                # Process alightings at this station
                if alightings > 0 and passengers_on_train:
                    #print(alightings, passengers_on_train)
                    # Select passengers to alight based on their boarding order
                    #alighting_passenger_ids = sorted(passengers_on_train)[:alightings]
                    try:
                        for _ in range(alightings):
                            # Randomly pick a passenger to alight
                            #print(passengers_on_train.items())
                            #print(list(passengers_on_train.items()))
                           
                            pid, passenger_info = random.choice(list(passengers_on_train.items()))
                            #print(pid, passenger_info)

                        #for pid in alighting_passenger_ids:
                        #   boarding_station, boarding_time = passengers_on_train[pid]
                            # Record the journey for the alighting passenger
                            passenger_journey = {
                                'PassengerID': pid,
                                'TrainID': train_id,
                                'BoardingStation': passenger_info['BoardingStation'],
                                'BoardingTime': passenger_info['BoardingTime'],
                                'AlightingStation': station,
                                'AlightingTime': timestamp,
                                'sBahnID': sbahn_id
                            }
                            passenger_paths.append(passenger_journey)

                            # Remove the passenger from the train
                            del passengers_on_train[pid]

                    except IndexError:  # this will handle only IndexError, don't use pure except
                        break 

                # Update the remaining passengers with the next possible alighting stations
                next_stations = self.get_next_stations(df, station, train_id)
                for pid in list(passengers_on_train):
                    if not next_stations:  # If there are no more stations, alight all remaining passengers
                        passenger_info = passengers_on_train[pid]
                        passenger_path = {
                            'PassengerID': pid,
                            'TrainID': train_id,
                            'BoardingStation': passenger_info['BoardingStation'],
                            'BoardingTime': passenger_info['BoardingTime'],
                            'AlightingStation': station,
                            'AlightingTime': timestamp
                            }
                        passenger_paths.append(passenger_path)
                        del passengers_on_train[pid]


        # Convert the list of passenger journeys to a DataFrame
        passenger_paths_df = pd.DataFrame(passenger_paths)
        # Save the DataFrame to a CSV file
        passenger_paths_df.to_csv('data/passenger_paths_new3.csv', index=True, sep=';', encoding='utf-8')

        return passenger_paths_df
    
    def route_mining(self, df):
        passenger_paths = []
        passenger_id = 1

        for train_id, train_group in df.groupby('TrainID'):
            passengers_on_train = {}  # Dictionary to track passengers (key: group_id, value: list of PassengerIDs)

            # Iterate over each station in the journey
            for index, row in train_group.iterrows():
                station = row['Station']
                timestamp = row['Arrival']
                boardings = row['Boardings']
                alightings = row['Alightings']
                sbahn_id = row ['sBahnID']


                # Assign PassengerIDs to each boarding passenger
                for _ in range(boardings):
                    if station not in passengers_on_train:
                        passengers_on_train[station] = []
                    passengers_on_train[station].append(passenger_id)
                    passenger_id += 1

                # Probabilistically determine alighting passengers
                if alightings > 0:
                    # Calculate probabilities for each passenger group
                    # (This requires a probabilistic model based on historical data or assumptions)
                    # For simplicity, let's assume equal probability for now
                    total_passengers = sum(len(group) for group in passengers_on_train.values())
                    if total_passengers > 0:
                        alighting_probs = {group_id: len(group) / total_passengers for group_id, group in passengers_on_train.items()}
                        alighting_passengers = random.choices(list(alighting_probs.keys()), weights=alighting_probs.values(), k=alightings)

                        for group_id in alighting_passengers:
                            # Select a passenger from the group to alight
                            if passengers_on_train[group_id]:
                                pid = passengers_on_train[group_id].pop(0)  # Pop the first passenger from the group
                                # Record the passenger's journey
                                passenger_path = {
                                    'PassengerID': pid,
                                    'TrainID': train_id,
                                    'BoardingStation': group_id,
                                    'BoardingTime': train_group.loc[train_group['Station'] == group_id, 'Arrival'].iloc[0],
                                    'AlightingStation': station,
                                    'AlightingTime': timestamp,
                                    'sBahnID': sbahn_id
                                    }
                                passenger_paths.append(passenger_path)
         # Convert the list of passenger journeys to a DataFrame
        passenger_paths_df = pd.DataFrame(passenger_paths)
        # Save the DataFrame to a CSV file
        passenger_paths_df.to_csv('data/passenger_paths_new.csv', index=False, sep=';', encoding='utf-8')

        return passenger_paths_df
    
    def preprocess_data(self, df):
        # Preprocess to create mappings for stations and timestamps for each TrainID
        station_map = df.groupby('TrainID')['Station'].apply(list)
        timestamp_map =  df.groupby('TrainID')['Arrival'].apply(lambda x: x.dt.strftime('%Y-%m-%d %H:%M:%S').tolist())#.apply(list)

        return station_map, timestamp_map
    
    def create_eventlog(self, df, routes):
        station_map, timestamp_map = self.preprocess_data(df)
        #print(timestamp_map)
        # Create a new DataFrame for the event log
        #event_log = pd.DataFrame(columns=['CaseID', 'Activity', 'Timestamp', 'sBahnID'])
        event_log = []
        unique_trips = routes[['TrainID', 'BoardingStation', 'AlightingStation', 'BoardingTime', 'AlightingTime', 'sBahnID']].drop_duplicates()
        #print(routes)
        for index, row in unique_trips.iterrows():
            train_id = row['TrainID']
            boarding_station = row['BoardingStation']
            alighting_station = row['AlightingStation']
            boarding_time = row['BoardingTime']
            alighting_time = row['AlightingTime']
            sbahnid = row['sBahnID']

            #ordered_stations = df[df['TrainID'] == train_id]['Station']
            ordered_stations = station_map[train_id]
            timestamp_order = timestamp_map[train_id]
            start_index = ordered_stations.index(boarding_station)#ordered_stations[ordered_stations == boarding_station]#.index[0]
            end_index = ordered_stations.index(alighting_station)#ordered_stations[ordered_stations == alighting_station]#.index[0]


            complete_route = ordered_stations[start_index:end_index + 1]

            #timestamp_order =  df[df['TrainID'] == train_id]['Arrival']#.astype(str).values.tolist()
            board_index = timestamp_order.index(boarding_time)#timestamp_order[timestamp_order == boarding_time].index[0]
            alight_index = timestamp_order.index(alighting_time)#timestamp_order[timestamp_order == alighting_time].index[0]

            journey_timestamps = timestamp_order[board_index:alight_index+1]


        #for index, journey in routes.iterrows():
           # passenger_id = journey['PassengerID']
           # train_id = journey['TrainID']
          #  boarding_station = journey['BoardingStation']
          #  alighting_station = journey['AlightingStation']
           # sbahnid = journey['sBahnID']
            #boarding_time = str(journey['BoardingTime'])
           # alighting_time = str(journey['AlightingTime'])
            #print(train_id, boarding_station, alighting_station, boarding_time,   alighting_time)

            # Get the ordered list of stations for the TrainID
            #station_order = df[df['TrainID'] == train_id]['Station'].values.tolist()
            #timestamp_order =  df[df['TrainID'] == train_id]['Arrival'].astype(str).values.tolist()
            #print(station_order)
            #print(timestamp_order)

            # Find the indices for the boarding and alighting stations
            #boarding_index = station_order.index(boarding_station)
            #alighting_index = station_order.index(alighting_station)

            #boarding_time_index = timestamp_order.index(boarding_time)
            #print(boarding_time_index)
            #arrival_time_index = timestamp_order.index(alighting_time)
            #print(arrival_time_index)

            # Create a list of stations the passenger would pass through
            #journey_stations = station_order[boarding_index:alighting_index]
            #journey_timestamps = timestamp_order[boarding_time_index:arrival_time_index]
    
            # Create a timestamp for each station in the journey (this part may need additional data or assumptions)
            # Here we just use the boarding time for simplicity
            #timestamps = [journey['BoardingTime']] * len(journey_stations)

             # Find all passengers who traveled this path
            passengers_on_this_trip = routes[(routes['TrainID'] == train_id) & 
                                        (routes['BoardingStation'] == boarding_station) & 
                                        (routes['AlightingStation'] == alighting_station)].drop_duplicates()
            
            # Assign this route to each passenger
            for passenger_id in passengers_on_this_trip:
                for station, timestamp in zip(complete_route, journey_timestamps):
                    event = {'CaseID': passenger_id, 'Activity': station, 'Timestamp': timestamp, 'sBahnID': sbahnid}
                    event_log.append(event)
                
    
            # Fill in the event log for this passenger
            #for station, timestamp in zip(journey_stations, journey_timestamps):
               # event = {'CaseID': passenger_id, 'Activity': station, 'Timestamp': timestamp, 'sBahnID': sbahnid}
               # event_log.append(event)
                #pd.concat([event_log, pd.DataFrame([event])], ignore_index=True)

            # Convert timestamp strings to actual datetime objects if necessary
        
        event_log = pd.DataFrame(event_log)
        event_log['Timestamp'] = pd.to_datetime(event_log['Timestamp'])
            # Save the event log to a CSV file
        event_log.to_csv('data/event_log.csv', index=True,  sep=';', encoding='utf-8')

            # Print the first few rows for verification
        print(event_log.head())

        return event_log


