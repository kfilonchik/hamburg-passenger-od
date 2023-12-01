from preprocessing.preprocessing import DataPreprocessing
from mining.routes import RoutesMining

def start_preprocessing(preprocessing):
    df = preprocessing.create_df()
    df = preprocessing.rename_columns(df)
    df = preprocessing.change_datatypes(df)

    return df

if __name__== '__main__':
    preprocessing = DataPreprocessing("data/sbahn_hamburg.csv")
    df = start_preprocessing(preprocessing)

    print(df)

    mining = RoutesMining(df)
    #routes = mining.route_mining_simple(df)
    routes= mining.read_eventlog("data/passenger_paths_new3.csv")
    eventlog = mining.create_eventlog(df, routes)
    

    



