import pandas as pd

# Load the datasets
sun_minute_df = pd.read_csv('./data/SUN Minute.csv')
dataset_df = pd.read_csv('./data/dataset.csv')

# Combine Date and Time columns in dataset_df to create a new column 'TmStamp'
# The Date and Time format is "15/02/2023 7:36:18"
dataset_df['TmStamp'] = pd.to_datetime(dataset_df['Date'] + ' ' + dataset_df['Time'], format='%d/%m/%Y %H:%M:%S').dt.floor('T')

# Convert TmStamp in sun_minute_df to datetime and floor to the minute
# The TmStamp format is "15/02/2023 7:36"
sun_minute_df['TmStamp'] = pd.to_datetime(sun_minute_df['TmStamp'], format='%d/%m/%Y %H:%M').dt.floor('T')

# Merge the two dataframes on the TmStamp column
merged_df = pd.merge(dataset_df, sun_minute_df, on='TmStamp', how='left')

# Save the merged dataframe to a new CSV file
merged_df.to_csv('./data/final_dataset.csv', index=False)

print("Merge complete.")
