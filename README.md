# Estimation individual travel routes using boarding and alighting counts within Hamburg City Train Network using Hidden Markov Model. 

### Dataset

Dataset was obtained from https://data.deutschebahn.com/dataset/passagierzahlung-s-bahn-hamburg/resource/480dc561-1417-4176-bbd6-8c60f01f6f47.html
The model was trained using only S1 data.

### Load data

Download the file and create 'data' folder, the name of the file should be sbahn_hamburg.csv.

First the data imputation methods should be applyed. For that the method preprocessing.enchace_data should be used.
If you have already the formatted data and no imputation method required, the rows 72/73 can be commented and df.create_formatted() method used.

### Start the programm:

python main.py

