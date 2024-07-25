import os
import sys
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
    df.df.drop(['date', 'time', 'DateTime','classroom_category','device_code','measured_rh','measured_co2', 'measured_pm1.0', 'measured_pm2.5', 'measured_pm10','grade','room_no','battv_min', 'batt24v_min','school_no','tracker2wm_avg'], axis=1, inplace=True)

    start_date = '2023-03-16'
    end_date = '2023-11-07'
    df.df = df.df[(df.df['tmstamp'] >= start_date) & (df.df['tmstamp'] <= end_date)]

    df.df = df.df.dropna(subset=['Measured T'])

    configs['cat_feats'] = []
    cat_feats = df.get_categorical_columns()
    cat_feats = list(set(cat_feats) - set(["tmstamp"]))
    for cat_feat in cat_feats:
        df.label_encoding(cat_feat)
        #print(f"Column: {cat_feat}, Uniques: {df.df[cat_feat].nunique()}")
        configs['cat_feats'].append(
            {
                "column_name" : cat_feat,
                "num_uniques": int(df.df[cat_feat].nunique())
            }
        )

    cont_feats = df.get_continuous_numeric_columns()
    cont_feats = list(set(cont_feats) - set(['device_code', TARGET_COLUMN]) - set(cat_feats))
    
    # no normalizing of preprocessing of the continious
    configs["num_feats"] = deepcopy(cont_feats)

    # Adding the target column to the list of continuous features for scaling later on
    cont_feats.append(TARGET_COLUMN)
    
    for col in df.get_nan_containing_columns():
        method = None
        if col in cont_feats:
            method = 'inter'
        elif col in cat_feats:
            method = "most_freq"
        else:
            print("Unknown column for the NaN fill: ", col)
            continue
        df.fillna(col, method=method)

    df.df = df.df.sort_values(by='TmStamp')
    df.df = df.df.reset_index(drop=True)
    
    df.slice_sequential(train_prop=TRAIN_SIZE, valid_prop=VALID_SIZE)

    # Apply min-max scaling to the numerical features 
    df.min_max_scale_data(cont_feats) #to avoid data leakage, so its after slice_sequential
    
    df.save_train_df()
    df.save_valid_df()
    df.save_test_df()

    configs["target_feature"] = TARGET_COLUMN
    write_config(os.path.join(output_dirpath, "configs.yaml"), configs)

if __name__ == "__main__":
    i_path = sys.argv[1]
    o_path = sys.argv[2]
    prepare_data(i_path, o_path)
