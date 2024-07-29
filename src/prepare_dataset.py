import os
import argparse
from typing import *
from loguru import logger
from copy import deepcopy
from utils import DataWrapper, write_config
import pandas as pd
import argparse

TARGET_COLUMN = "measured_t_mean"

def prepare_data(data_path: str,
                 output_dirpath: str,
                 train_prop: float = 0.7,
                 valid_prop: float = 0.1,
                 scale_method: Optional[str] = None):
    logger.info(f"Args:  {data_path} ; {output_dirpath} ; {train_prop} ; {valid_prop} ; {scale_method}")
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

    df.df.dropna(subset=[TARGET_COLUMN])

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
    
    logger.info(f"Removing the `measured_t_*` columns, except : {TARGET_COLUMN}")
    for feat in df.df.columns:
        if 'measured_t_' in feat and feat != TARGET_COLUMN:
            df.df.drop(labels=[feat], axis=1, inplace=True)
        elif ('_min' in feat or '_max' in feat) and feat != TARGET_COLUMN:
            df.df.drop(labels=[feat], axis=1, inplace=True)
    
    cont_feats = df.get_continuous_numeric_columns()
    date_time=['Day','Minute','Hour']
    cont_feats = list(set(cont_feats) - set(['device_code', TARGET_COLUMN]) - set(cat_feats)  - set(date_time))
        
    # no normalizing of preprocessing of the continious
    configs["num_feats"] = deepcopy(cont_feats)

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
    df.slice_sequential(train_prop=train_prop, valid_prop=valid_prop)
    logger.info(f"Sliced the data into train, valid, test splits with the proportion: ", 
                train_prop, valid_prop, round(1. - train_prop - valid_prop, 2))
    # Apply scaling to the numerical features
    if not scale_method is None:
        logger.info(f"Appliing the {scale_method} to the dataset")
        if 'standardize' == scale_method:
            scaler_data_path = df.standardize_data(cont_feats, TARGET_COLUMN)
        elif 'min_max' == scale_method:
            scaler_data_path = df.min_max_scale_data(cont_feats, TARGET_COLUMN)  # to avoid data leakage, so its after slice_sequential
        else:
            logger.error(f"Wrong given option for scaling: {scale_method}, choose from [`standardize`, `min_max`]")
            raise ValueError(scale_method)
        configs['scaler_data_path'] = scaler_data_path
        assert os.path.exists(configs['scaler_data_path'])
    df.save_train_df()
    if valid_prop > 0.:
        df.save_valid_df()
    df.save_test_df()

    configs["target_feature"] = TARGET_COLUMN
    write_config(os.path.join(output_dirpath, "configs.yaml"), configs)
    logger.info(f"Data and assets are saved in the : {output_dirpath}")
    logger.info(f"Data splits have this sizes each\n \
        Train: {df.train_df.shape[0]}, \
        Valid: {df.valid_df.shape[0]}, \
        Test : {df.test_df.shape[0]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-path', help="Path to the input dataset", type=str, required=True)
    parser.add_argument('-o', '--output-dirpath', help="Path to the output directory", type=str, required=True)
    parser.add_argument('-t', '--train-prop', help="Train proportion in data [0;1]", type=float, required=False, default=0.8)
    parser.add_argument('-v', '--valid-prop', help="Valid proportion in data [0;1]", type=float, required=False, default=0.2)
    parser.add_argument('-s', '--scale-method', help="Scale method of data, optional", type=str, required=False, default=None, choices=['standardize', 'min_max'])
    args = parser.parse_args()
    prepare_data(args.data_path, args.output_dirpath, args.train_prop, args.valid_prop, args.scale_method)
