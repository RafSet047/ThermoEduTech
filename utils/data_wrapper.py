import os
import sys
from typing import *
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from scipy.stats import pointbiserialr
from utils.utils import write_json

class DataWrapper:
    """
    A wrapper class for handling data operations on a pandas DataFrame.

    Attributes:
        df (pd.DataFrame): The DataFrame to perform operations on.
        output_dir (str): Directory to save output data and statistics. If not provided, defaults to a 'data' folder in the current working directory.
    """
    def __init__(self, df: Union[pd.DataFrame, str], output_dir: Optional[str] = None):
        """
        Initializes the DataWrapper with a DataFrame and an optional output directory.

        Args:
            df (pd.DataFrame): The DataFrame to perform operations on.
            output_dir (Optional[str]): Directory to save output data and statistics. If not provided, defaults to a 'data' folder in the current working directory.
        """
        self._df = None
        if isinstance(df, str):
            print(df)
            self._df = pd.read_csv(df)
        else:
            self._df = df

        self.output_dir = output_dir

        self._train_df = None
        self._valid_df = None
        self._test_df = None

        if self.output_dir is None:
            self.output_dir = os.path.join(os.getcwd(), 'dataset', 'processed')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'assets'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'assets', 'labels'), exist_ok=True)
        self.__preprocess_columns()

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @df.setter
    def df(self, new_df: pd.DataFrame) -> None:
        self._df = new_df

    def __preprocess_columns(self):
        cols = self._df.columns
        cols = [col.replace(" ", "_").lower() for col in cols]
        self._df.columns = cols
    
    @staticmethod
    def calculate_ma(x: pd.Series, period: int) -> pd.Series:
        """
        Calculates the moving average (MA) for a given pandas Series.

        Args:
            x (pd.Series): The input data series.
            period (int): The period over which to calculate the moving average.

        Returns:
            pd.Series: The moving average of the input series.
        """
        return x.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(x: pd.Series, period: int) -> pd.Series:
        """
        Calculates the exponential moving average (EMA) for a given pandas Series.

        Args:
            x (pd.Series): The input data series.
            period (int): The period over which to calculate the exponential moving average.

        Returns:
            pd.Series: The exponential moving average of the input series.
        """
        return x.ewm(span=period, adjust=True).mean()

    @staticmethod
    def standardize(x: Union[np.array, pd.Series], scaler: Optional[StandardScaler] = None) -> Tuple[Union[np.array, pd.Series], StandardScaler]:
        """
        Standardizes the input data by removing the mean and scaling to unit variance.

        Args:
            x (Union[np.array, pd.Series]): The input data series or array.
            scaler (StandardScaler): The standard scaler for standardization. If None, it is calculated from the data.

        Returns:
            Tuple[Union[np.array, pd.Series], scaler]: A tuple containing the standardized data and a scaler.
        """
        if isinstance(x, pd.Series):
            x = x.values.reshape(-1, 1)
        if scaler is None:
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)
            return x_scaled, scaler
        return scaler.transform(x), scaler
    
    @staticmethod
    def min_max_scale(x: Union[np.array, pd.Series], scaler: Optional[MinMaxScaler] = None) -> Tuple[Union[np.array, pd.Series], MinMaxScaler]:
        """
        Scales the input data to a given range [0, 1] using min-max scaling.

        Args:
            x (Union[np.array, pd.Series]): The input data series or array.
            scaler (Optional[MinMaxScaler]): The min max scaler. If None, it is calculated from the data.

        Returns:
            Tuple[Union[np.array, pd.Series], MinMaxScaler]: A tuple containing the scaled data and a scaler.
        """
        if isinstance(x, pd.Series):
            x = x.values.reshape(-1, 1)
        if scaler is None:
            scaler = MinMaxScaler()
            x_scaled = scaler.fit_transform(x)
            return x_scaled, scaler
        return scaler.transform(x), scaler

    def slice_sequential(self, train_prop: float = 0.7, valid_prop: float = 0.1):
        """
        Splits the DataFrame into training, validation, and test sets sequentially based on provided proportions.

        Args:
            train_prop (float): Proportion of the data to be used for training. Default is 0.7.
            valid_prop (float): Proportion of the data to be used for validation. Default is 0.1.

        The remaining data after allocating the training and validation sets is used for the test set.
        """
        data_size = self._df.shape[0]

        train_size = int(data_size * train_prop)
        valid_size = int(data_size * valid_prop)

        self._train_df = self._df.iloc[:train_size, :].copy()
        self._valid_df = self._df.iloc[train_size:train_size+valid_size, :].copy()
        self._test_df = self._df.iloc[train_size+valid_size:, :].copy()

    def standardize_data(self, x_columns: List[str], y_column: str):
        """
        Standardizes the training, validation, and test DataFrames using the mean and standard deviation of the training data.
        
        The mean and standard deviation are calculated from the training data and then applied to the validation and test data to standardize them.
        The scaler is also saved as binary file in the 'assets' directory within the output directory.
        """
        self._train_df[x_columns], x_scaler = DataWrapper.standardize(self._train_df[x_columns])
        self._valid_df[x_columns], _ = DataWrapper.standardize(self._valid_df[x_columns], x_scaler)
        self._test_df[x_columns], _ = DataWrapper.standardize(self._test_df[x_columns], x_scaler)

        x_scaler_path = os.path.join(self.output_dir, 'assets', 'standard_scaler_x.joblib')
        joblib.dump(x_scaler, x_scaler_path)

        self._train_df[y_column], y_scaler = DataWrapper.standardize(self._train_df[y_column])
        self._valid_df[y_column], _ = DataWrapper.standardize(self._valid_df[y_column], y_scaler)
        self._test_df[y_column], _ = DataWrapper.standardize(self._test_df[y_column], y_scaler)

        y_scaler_path = os.path.join(self.output_dir, 'assets', 'standard_scaler_y.joblib')
        joblib.dump(y_scaler, y_scaler_path)

        scaler_data = {
            "x_scaler_path" : x_scaler_path,
            "y_scaler_path" : y_scaler_path,
            "x_columns" : x_columns,
            "y_column" : y_column 
        }
        scaler_data_path = os.path.join(self.output_dir, 'assets', 'standard_scaler_data.json')
        write_json(scaler_data_path, scaler_data)
        return scaler_data_path

    def min_max_scale_data(self, x_columns: List[str], y_column: str):
        """
        Scales the training, validation, and test DataFrames using min-max scaling with the minimum and maximum values of the training data.
        
        The minimum and maximum values are calculated from the training data and then applied to the validation and test data to scale them.
        The scaler is also saved as CSV files in the 'stats' directory within the output directory.
        """
        self._train_df[x_columns], x_scaler = DataWrapper.min_max_scale(self._train_df[x_columns])
        self._valid_df[x_columns], _ = DataWrapper.min_max_scale(self._valid_df[x_columns], x_scaler)
        self._test_df[x_columns], _ = DataWrapper.min_max_scale(self._test_df[x_columns], x_scaler)

        x_scaler_path = os.path.join(self.output_dir, 'assets', 'min_max_scaler_x.joblib')
        joblib.dump(x_scaler, x_scaler_path)

        self._train_df[y_column], y_scaler = DataWrapper.min_max_scale(self._train_df[y_column])
        self._valid_df[y_column], _ = DataWrapper.min_max_scale(self._valid_df[y_column], y_scaler)
        self._test_df[y_column], _ = DataWrapper.min_max_scale(self._test_df[y_column], y_scaler)

        y_scaler_path = os.path.join(self.output_dir, 'assets', 'min_max_scaler_y.joblib')
        joblib.dump(y_scaler, y_scaler_path)

        scaler_data = {
            "x_scaler_path" : x_scaler_path,
            "y_scaler_path" : y_scaler_path,
            "x_columns" : x_columns,
            "y_column" : y_column 
        }
        scaler_data_path = os.path.join(self.output_dir, 'assets', 'min_max_scaler_data.json')
        write_json(scaler_data_path, scaler_data)
        return scaler_data_path

    @staticmethod
    def inverse_rescale_data(x_scaler_path: str, y_scaler_path: str, data: pd.DataFrame, x_columns: List[str], y_column: str) -> pd.DataFrame:
        assert os.path.exists(x_scaler_path)
        assert os.path.exists(y_scaler_path)
        df = data.copy()
        x_scaler = joblib.load(x_scaler_path)
        y_scaler = joblib.load(y_scaler_path)

        y = df[y_column].values.reshape(-1, 1)

        df[x_columns] = x_scaler.inverse_transform(df[x_columns])
        df[y_column] = y_scaler.inverse_transform(y)
        return df

    def remove_outliers(self, column: str):
        Q1 = self._df[column].quantile(0.25)
        Q3 = self._df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        self._df = self._df[(self._df[column] >= lower_bound) & (self._df[column] <= upper_bound)]

    @staticmethod
    def get_windowed_data(x: Union[np.array, list, pd.Series], window_size: int, horizon: int) -> pd.DataFrame:
        """
        This function creates time series dataset by given window size and horizon.
        Funciton creates the window_size overlapping arrays.
        
        Args:
            x           : data series
            window_size : length of overlapping slices
            horizon     : target variable length
        Returns:
            df : dataframe of time series
        
        E.g:
           Given series by indexes
           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, ....]
           
           By window_size=7, horizon=1

           The output array by indexes will be
           array([[ 0,  1,  2,  3,  4,  5,  6,  7],
                  [ 1,  2,  3,  4,  5,  6,  7,  8],
                  [ 2,  3,  4,  5,  6,  7,  8,  9],
                  [ 3,  4,  5,  6,  7,  8,  9, 10],
                  ...
                  ])
            
        """
        cols = []
        for i in range(window_size):
            cols.append(f'x_{i}')
        for j in range(horizon):
            cols.append(f'y_{j}')
        # Create one row of steps as nested array
        window_steps = np.expand_dims(np.arange(window_size + horizon), axis=0)
        # Create transposed matrix of indexes
        indexes = window_steps + np.expand_dims(np.arange(len(x) - (window_size + horizon - 1)), axis=0).T
        # Creating pandas dataframe with indexes of x then putting columns
        df = pd.DataFrame(np.array(x)[indexes])
        df.columns = cols
        return df

    @staticmethod
    def get_return(x: pd.Series):
        """
        Calculates the percentage change (return) of a given pandas Series.

        Args:
            x (pd.Series): The input data series.

        Returns:
            pd.Series: The percentage change of the input series, with NaN values filled with 0.
        """
        x = x.pct_change(1)
        x.fillna(0., inplace=True)
        return x
    
    @staticmethod
    def get_difference(x: pd.Series):
        """
        Calculates the difference between consecutive elements in a given pandas Series.

        Args:
            x (pd.Series): The input data series.

        Returns:
            pd.Series: The difference between consecutive elements of the input series, with NaN values filled with 0.
        """
        d = x - x.shift(1)
        d.fillna(0., inplace=True)
        return d

    @property
    def train_df(self):
        return self._train_df

    def save_train_df(self):
        self._train_df.to_csv(os.path.join(self.output_dir, 'train.csv'), index=False)

    @property
    def valid_df(self):
        return self._valid_df

    def save_valid_df(self):
        self._valid_df.to_csv(os.path.join(self.output_dir, 'valid.csv'), index=False)

    @property
    def test_df(self):
        return self._test_df

    def save_test_df(self):
        self._test_df.to_csv(os.path.join(self.output_dir, 'test.csv'), index=False)

    @staticmethod
    def create_rolling_windows(data: pd.DataFrame, window_size: int, step_size: int):
        """
        Splits the data into rolling windows for time series cross-validation.

        Args:
            data (pd.DataFrame): dataset which should be sliced into windowed subsets
            window_size (int): Size of each rolling window.
            step_size (int): Step size to move the window.

        Returns:
            List[Tuple[pd.DataFrame, pd.DataFrame]]: List of training and validation DataFrames.
        """
        windows = []
        for start in range(0, len(data) - window_size, step_size):
            end = start + window_size
            train = data.iloc[start:end]
            valid = data.iloc[end:end + step_size]
            windows.append((train, valid))
        return windows

    def remove_by_dates(self, column: str, start_date: str, end_date: str) -> None:

        self._df[column] = pd.to_datetime(self._df[column], format='%d/%m/%Y %H:%M')
        start_date = pd.to_datetime(start_date, format='%d/%m/%Y')
        end_date = pd.to_datetime(end_date, format='%d/%m/%Y') 
        self._df = self._df[(self._df[column] >= start_date) & (self._df[column] <= end_date)]

    def sort_by_values(self, column: str) -> None:
        self._df.sort_values(by=column, inplace=True)
        self._df.reset_index(drop=True, inplace=True)
        
    def fillna(self, col_name: str, method: str):
        if "zeroes" == method:
            self._df[col_name].fillna(0, inplace=True)
        elif "mean" == method:
            self._df[col_name].fillna(self._df[col_name].dropna().mean(), inplace=True)
        elif "median" == method:
            self._df[col_name].fillna(self._df[col_name].dropna().median(), inplace=True)
        elif "most_freq" == method:
            self._df[col_name].fillna(self._df[col_name].dropna().mode().values[0], inplace=True)
        elif "inter" == method:
            self._df[col_name].interpolate(method='cubic', inplace=True) 
        elif "ffill" == method:
            self._df[col_name].fillna(method='ffill', inplace=True)
        else:
            NotImplementedError(f"Not implemented inputation type: {method}")

    def one_hot_encoding(self, col_name: Union[str, List[str]]):
        """
        One-hot encodes specified categorical columns in the DataFrame.

        Args:
            columns (List[str]): List of column names to be one-hot encoded.
        
        Updates the DataFrame in place with one-hot encoded columns and retains other columns.
        """
        if isinstance(col_name, str):
            col_name = [col_name]
        self._df = pd.get_dummies(self._df, columns=col_name, drop_first=True, dtype=float)
        
    def label_encoding(self, col_name: str):
        """
        Label encodes a specified categorical column in the DataFrame.

        Args:
            column (str): The name of the column to be label encoded.

        Updates the DataFrame in place with the label encoded column.
        """
        le = LabelEncoder()
        self._df[col_name] = le.fit_transform(self._df[col_name])

        joblib.dump(le, os.path.join(self.output_dir, 'assets', 'labels', f"{col_name}.joblib"))

    def get_numeric_columns(self) -> List[str]:
        """
        Identifies and returns a list of column names that are numeric (continuous data).

        Returns:
            List[str]: List of numeric column names.
        """
        return self._df.select_dtypes(include=['number']).columns.tolist()

    def get_categorical_columns(self) -> List[str]:
        """
        Identifies and returns a list of column names that are categorical.

        Returns:
            List[str]: List of categorical column names.
        """
        return self._df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def get_other_columns(self) -> List[str]:
        """
        Identifies and returns a list of column names that are neither categorical nor numeric.

        Returns:
            List[str]: List of other column names.
        """
        categorical_cols = self.get_categorical_columns()
        numeric_cols = self.get_numeric_columns()
        all_cols = set(self._df.columns)
        other_cols = all_cols - set(categorical_cols) - set(numeric_cols)
        return list(other_cols)

    def get_continuous_numeric_columns(self) -> List[str]:
        """
        Identifies and returns a list of column names that are continuous numeric.

        Returns:
            List[str]: List of continuous numeric column names.
        """
        numeric_cols = self.get_numeric_columns()
        continuous_cols = [col for col in numeric_cols if self._df[col].nunique() > 10]
        return continuous_cols

    def get_binary_numeric_columns(self) -> List[str]:
        """
        Identifies and returns a list of column names that are binary numeric.

        Returns:
            List[str]: List of binary numeric column names.
        """
        numeric_cols = self.get_numeric_columns()
        binary_cols = [col for col in numeric_cols if self._df[col].nunique() == 2]
        return binary_cols

    def get_k_categorical_numeric_columns(self, k: int) -> List[str]:
        """
        Identifies and returns a list of column names that are binary numeric.

        Args:
            k [int]: maximum number of unique values within the column that should be considered as categorical
        Returns:
            List[str]: List of binary numeric column names.
        """
        numeric_cols = self.get_numeric_columns()
        cols = [col for col in numeric_cols if self._df[col].nunique() <= k]
        return cols

    def describe(self, column: str) -> pd.Series:
        """
        Returns the detailed describtion of the specified column in the self._df

        Args:
            column (str): column name

        Returns:
            pd.Series: description of column
        """
        return self._df[column].describe()

    def value_counts(self, column: str) -> pd.Series:
        """
        Returns the Series where the key is the unique value within the column and the value is its count

        Args:
            column (str): column name

        Returns:
            pd.Series: counter of unique values
        """
        return self._df[column].value_counts()

    def get_nan_containing_columns(self) -> List[str]:
        """
        Returns list of column names where is at least one NaN value

        Returns:
            List[str]: List of columns
        """
        return self._df.columns[self._df.isna().any()].tolist()

    def cont_corr(self, columns: List[str] = []) -> pd.Series:
        return self._df[columns].corr()
    
    def binary_corr(self, column_x: str, column_y: str) -> float:
        corr, _ = pointbiserialr(x=self._df[column_x], y=self._df[column_y])
        return corr

if __name__ == "__main__":
    print("ok")
