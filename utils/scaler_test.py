import os
import joblib
import numpy as np
import pandas as pd
from data_wrapper import DataWrapper
from utils import load_json

def test_scaler():
    output_path = './utils/tmp'
    num_cols = 10
    data = np.random.randn(100, num_cols)
    
    cols = [f'x_{i}' for i in range(num_cols - 1)]
    cols.append('y')
    assert len(cols) == num_cols
    
    df = DataWrapper(df=pd.DataFrame(data=data, columns=cols), output_dir=output_path)
    
    df.slice_sequential(train_prop=0.7, valid_prop=0.2)
    
    df_train_unscaled = df.train_df.copy()
    df_valid_unscaled = df.valid_df.copy()
    df_test_unscaled  = df.test_df.copy()
    
    scaler_data_path = df.min_max_scale_data(x_columns=cols[:-1], y_column=cols[-1])
    scaler_data = load_json(scaler_data_path)

    df_train_scaled = df.train_df.copy()
    df_valid_scaled = df.valid_df.copy()
    df_test_scaled  = df.test_df.copy()

    df_train_rescaled = DataWrapper.inverse_rescale_data(scaler_data['x_scaler_path'], scaler_data['y_scaler_path'], 
                                                        df_train_scaled,
                                                        scaler_data['x_columns'],
                                                        scaler_data['y_column'])
    df_valid_rescaled = DataWrapper.inverse_rescale_data(scaler_data['x_scaler_path'], scaler_data['y_scaler_path'], 
                                                        df_valid_scaled,
                                                        scaler_data['x_columns'],
                                                        scaler_data['y_column'])
    df_test_rescaled = DataWrapper.inverse_rescale_data(scaler_data['x_scaler_path'], scaler_data['y_scaler_path'], 
                                                        df_test_scaled,
                                                        scaler_data['x_columns'],
                                                        scaler_data['y_column'])
    print("Original and Inversed difference in TRAIN : ", (df_train_unscaled - df_train_rescaled).mean())
    print("Original and Inversed difference in VALID : ", (df_valid_unscaled - df_valid_rescaled).mean())
    print("Original and Inversed difference in TEST  : ", (df_test_unscaled - df_test_rescaled).mean())

if __name__ == "__main__":
    test_scaler()