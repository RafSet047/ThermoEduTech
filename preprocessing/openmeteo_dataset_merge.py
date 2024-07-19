import pandas as pd

# Load the datasets
final_dataset_path = '../data/final_dataset.csv'
krigeville_openmeteo_path = '../data/Krigeville_OpenMeteo.csv'
mostertsdrift_openmeteo_path = '../data/Mostertsdrift_OpenMeteo.csv'

final_dataset = pd.read_csv(final_dataset_path)
krigeville_openmeteo = pd.read_csv(krigeville_openmeteo_path)
mostertsdrift_openmeteo = pd.read_csv(mostertsdrift_openmeteo_path)

# Combine 'Data' and 'Time' columns in final_dataset to create a new column 'TmStamp' and floor to the hour
final_dataset['TmStamp'] = pd.to_datetime(final_dataset['Date'] + ' ' + final_dataset['Time'], format='%d/%m/%Y %H:%M:%S').dt.floor('H')

# Convert 'date' to datetime and floor to the hour
krigeville_openmeteo['date'] = pd.to_datetime(krigeville_openmeteo['date'], utc=True).dt.floor('H')
mostertsdrift_openmeteo['date'] = pd.to_datetime(mostertsdrift_openmeteo['date'], utc=True).dt.floor('H')


# Make sure both datetime columns are timezone-aware with the same timezone
final_dataset['TmStamp'] = final_dataset['TmStamp'].dt.tz_localize('UTC')


# Filter the dataset where School No equals 1 and merge with Krigeville data
filtered_final_dataset_1 = final_dataset[final_dataset['School No'] == 1]
merged_data_1 = pd.merge(
    filtered_final_dataset_1,
    krigeville_openmeteo,
    left_on='TmStamp',
    right_on='date',
    suffixes=('_final', '_openmeteo')
)

# Filter the dataset where School No equals 2 and merge with Mostertsdrift data
filtered_final_dataset_2 = final_dataset[final_dataset['School No'] == 2]
merged_data_2 = pd.merge(
    filtered_final_dataset_2,
    mostertsdrift_openmeteo,
    left_on='TmStamp',
    right_on='date',
    suffixes=('_final', '_openmeteo')
)

# Combine the merged datasets
combined_merged_data = pd.concat([merged_data_1, merged_data_2])

# Save the updated dataset to a new CSV file
output_path = '../data/finaldataset_openmeteo.csv'
combined_merged_data.to_csv(output_path, index=False)

print("Merged dataset saved to:", output_path)
