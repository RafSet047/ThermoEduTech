import os
import argparse
from loguru import logger
from copy import deepcopy
from utils import DataWrapper, write_config
import pandas as pd

TRAIN_SIZE = 0.7
VALID_SIZE = 0.1

TARGET_COLUMN = "measured_t"


def prepare_data(data_path: str, output_dirpath: str):
    configs = {}
    df = DataWrapper(data_path, output_dirpath)

    # Convert 'Date' and 'Time' columns to datetime
    df.df['DateTime'] = pd.to_datetime(df.df['date'] + ' ' + df.df['time'], format='%d/%m/%Y %H:%M:%S')

    # Extract date and time features
    df.df['Month'] = df.df['DateTime'].dt.month
    df.df['Day'] = df.df['DateTime'].dt.day
    df.df['Hour'] = df.df['DateTime'].dt.hour
    df.df['Minute'] = df.df['DateTime'].dt.minute

    # Drop columns
    df.df.drop(['date', 'time', 'DateTime', 'datetime','outdoor_temperature_min','outdoor_temperature_mean','outdoor_temperature_max','classroom_category', 'device_code', 'measured_rh_min','measured_rh_mean','measured_rh_max', 'measured_co2_min','measured_co2_mean','measured_co2_max',
                'measured_pm1.0_min','measured_pm1.0_mean','measured_pm1.0_max', 'measured_pm2.5_min','measured_pm2.5_mean','measured_pm2.5_max', 'measured_pm10_min','measured_pm10_mean','measured_pm10_max', 'grade', 'room_no', 'battv_min_min', 'battv_min_mean','battv_min_max','batt24v_min_min','batt24v_min_mean','batt24v_min_max',
                'school_no', 'tracker2wm_avg_min','tracker2wm_avg_mean','tracker2wm_avg_max'], axis=1, inplace=True)
    logger.info("Dropped the unnecessary columns")

    # removing by date
    start_date = '16/03/2023'
    end_date = '07/11/2023'
    logger.info(f"Dataset size BEFORE dropping the rows by date: {df.df.shape[0]}")
    df.remove_by_dates('tmstamp', start_date, end_date)
    logger.info(f"Dataset size AFTER dropping the rows by date: {df.df.shape[0]}")

    df.df.dropna(subset=['measured_t'])

    df.df.sort_values(by='tmstamp')

    configs['cat_feats'] = []
    cat_feats = df.get_categorical_columns()
    cat_feats = list(set(cat_feats) - set(["tmstamp"]))
    for cat_feat in cat_feats:
        df.label_encoding(cat_feat)
        configs['cat_feats'].append(
            {
                "column_name": cat_feat,
                "num_uniques": int(df.df[cat_feat].nunique())
            }
        )
    logger.info(f"Overall encoded {len(cat_feats)} cateforical features")
    logger.info(cat_feats)
    
    cont_feats = df.get_continuous_numeric_columns()
    date_time=['Day','Minute','Hour']
    cont_feats = list(set(cont_feats) - set(['device_code', TARGET_COLUMN]) - set(cat_feats)  - set(date_time))

    # no normalizing of preprocessing of the continious
    configs["num_feats"] = deepcopy(cont_feats)

    # Adding the target column to the list of continuous features for scaling later on
    cont_feats.append(TARGET_COLUMN)

    for col in df.get_nan_containing_columns():
        method = None
        if col in cont_feats:
            method = 'ffill'
        elif col in cat_feats:
            method = "most_freq"
        else:
            print("Unknown column for the NaN fill: ", col)
            continue
        df.fillna(col, method=method)
    logger.info("Filled the missing values of variables")
    df.slice_sequential(train_prop=TRAIN_SIZE, valid_prop=VALID_SIZE)
    logger.info(f"Sliced the data into train, valid, test splits with the proportion: ", 
                TRAIN_SIZE, VALID_SIZE, round(1. - TRAIN_SIZE - VALID_SIZE, 2))
    # Apply min-max scaling to the numerical features
    df.min_max_scale_data(cont_feats)  # to avoid data leakage, so its after slice_sequential
    logger.info("Applied the MinMaxScaler to the dataset")
    logger.info(cont_feats)
    
    df.save_train_df()
    df.save_valid_df()
    df.save_test_df()

    configs["target_feature"] = TARGET_COLUMN
    write_config(os.path.join(output_dirpath, "configs.yaml"), configs)
    logger.info("Data and assets are saved in the : ", output_dirpath)
    logger.info(f"Data splits have this sizes each\n \
        Train: {df.train_df.shape[0]}, \
        Valid: {df.valid_df.shape[0]}, \
        Test : {df.test_df.shape[0]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-path', help="Path to the input dataset", type=str, required=True)
    parser.add_argument('-o', '--output-dirpath', help="Path to the output directory", type=str, required=True)
    args = parser.parse_args()
    prepare_data(args.data_path, args.output_dirpath)
