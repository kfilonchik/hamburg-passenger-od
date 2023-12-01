import pandas as pd


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

        return dataset


    


    


