import seaborn as sns
from preprocessing import DataPreprocessing
import matplotlib.pyplot as plt

preprocessing = DataPreprocessing("data/sbahn_hamburg.csv")
df = preprocessing.create_df()
df = preprocessing.rename_columns(df)
df = preprocessing.change_datatypes(df)
df = df.loc[df['sBahnID'] == 'S1']

# Assuming df_corrected has a datetime column for the date of each observation
# If the date is part of the 'dtmIstAnkunftDatum' and in the format 'dd.mm.yyyy hh:mm',
# we need to extract the date part and convert it to datetime format


df['Hour'] = df['Arrival'].dt.hour

# Creating the scatter plot

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Boardings', y='Alightings', data=df)
plt.title('Scatter Plot of Boardings vs. Alightings')
plt.xlabel('Boarding Counts')
plt.ylabel('Alighting Counts')
plt.show()


# Preparing the data for heat map
# Aggregating data by station and hour
df_heat_map_agg = df.groupby(['Station', 'Hour']).agg({'Boardings': 'sum', 'Alightings': 'sum'}).reset_index()

# Pivoting the DataFrame for seaborn heatmap
heatmap_data = df_heat_map_agg.pivot("Station", "Hour", "Boardings")  # Example for Boardings

# Creating the heat map
plt.figure(figsize=(15, 8))
sns.heatmap(heatmap_data, #annot=True, #fmt="d", 
            cmap="YlGnBu")
plt.title('Heat Map of Boarding Counts by Station and Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Station')
plt.show()


# Aggregating data by station
df_station_aggregated = df.groupby('Station').sum().reset_index()

# Melting the DataFrame for seaborn barplot
df_station_melted = df_station_aggregated.melt('Station', var_name='Type', value_name='Counts')

# Creating the bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x='Station', y='Counts', hue='Type', data=df_station_melted)
plt.title('Station-wise Boarding and Alighting Counts')
plt.xlabel('Station')
plt.xticks(rotation=45)
plt.ylabel('Total Counts')
plt.show()


# Plotting histograms

df = df.loc[df['sBahnID'] == 'S1']

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['Boardings'], kde=False, color='blue')
plt.title('Histogram of Boardings')
plt.xlabel('Number of Boardings')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(df['Alightings'], kde=False, color='green')
plt.title('Histogram of Alightings')
plt.xlabel('Number of Alightings')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
