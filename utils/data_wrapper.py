import os
import sys
from typing import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pointbiserialr

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
        os.system(f"mkdir {os.path.join(self.output_dir, 'stats')}")

        self.__preprocess_columns()

    @property
    def df(self):
        return self._df

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
    def standardize(x: Union[np.array, pd.Series], mean: Optional[float] = None, std: Optional[float] = None) -> Tuple[Union[np.array, pd.Series], Tuple[float, float]]:
        """
        Standardizes the input data by removing the mean and scaling to unit variance.

        Args:
            x (Union[np.array, pd.Series]): The input data series or array.
            mean (Optional[float]): The mean value for standardization. If None, it is calculated from the data.
            std (Optional[float]): The standard deviation for standardization. If None, it is calculated from the data.

        Returns:
            Tuple[Union[np.array, pd.Series], Tuple[float, float]]: A tuple containing the standardized data and a tuple of the mean and standard deviation used.
        """
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std()
        return (x - mean) / (std + 1e-6), (mean, std)
    
    @staticmethod
    def min_max_scale(x: Union[np.array, pd.Series], min_: Optional[float] = None, max_: Optional[float] = None) -> Tuple[Union[np.array, pd.Series], Tuple[float, float]]:
        """
        Scales the input data to a given range [0, 1] using min-max scaling.

        Args:
            x (Union[np.array, pd.Series]): The input data series or array.
            min_ (Optional[float]): The minimum value for scaling. If None, it is calculated from the data.
            max_ (Optional[float]): The maximum value for scaling. If None, it is calculated from the data.

        Returns:
            Tuple[Union[np.array, pd.Series], Tuple[float, float]]: A tuple containing the scaled data and a tuple of the minimum and maximum values used.
        """
        if min_ is None:
            min_ = x.min()
        if max_ is None:
            max_ = x.max()
        return (x - min_) / (max_ - min_ + 1e-6), (min_, max_)

    @staticmethod
    def scale_quantiles(x: Union[np.array, pd.Series], median: Optional[float] = None, q25: Optional[float] = None, q75: Optional[float] = None) -> Tuple[Union[np.array, pd.Series], Tuple[float, float]]:
        """
        Scales the input data using the median and interquartile range (IQR).

        Args:
            x (Union[np.array, pd.Series]): The input data series or array.
            median (Optional[float]): The median value for scaling. If None, it is calculated from the data.
            q25 (Optional[float]): The 25th percentile (first quartile) value for scaling. If None, it is calculated from the data.
            q75 (Optional[float]): The 75th percentile (third quartile) value for scaling. If None, it is calculated from the data.

        Returns:
            Tuple[Union[np.array, pd.Series], Tuple[float, float]]: A tuple containing the scaled data and a tuple of the median, 25th percentile, and 75th percentile values used.
        """
        if median is None:
            median = np.median(x)
        if q25 is None: 
            q25 = np.quantile(x, 25)
        if q75 is None:
            q75 = np.quantile(x, 75)
        return (x - median) / (q75 - q25), (median, q25, q75)

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

    def standardize_data(self, columns = List[str]):
        """
        Standardizes the training, validation, and test DataFrames using the mean and standard deviation of the training data.
        
        The mean and standard deviation are calculated from the training data and then applied to the validation and test data to standardize them.
        The mean and standard deviation are also saved as CSV files in the 'stats' directory within the output directory.
        """
        self._train_df[columns], (train_mean, train_std) = DataWrapper.standardize(self._train_df[columns])
        self._valid_df[columns], _ = DataWrapper.standardize(self._valid_df[columns], train_mean, train_std)
        self._test_df[columns], _ = DataWrapper.standardize(self._test_df[columns], train_mean, train_std)

        train_mean.to_csv(os.path.join(self.output_dir, 'stats', 'mean.csv'))
        train_std.to_csv(os.path.join(self.output_dir, 'stats', 'std.csv'))

    def min_max_scale_data(self, columns: List[str]):
        """
        Scales the training, validation, and test DataFrames using min-max scaling with the minimum and maximum values of the training data.
        
        The minimum and maximum values are calculated from the training data and then applied to the validation and test data to scale them.
        The minimum and maximum values are also saved as CSV files in the 'stats' directory within the output directory.
        """
        self._train_df[columns], (train_min, train_max) = DataWrapper.min_max_scale(self._train_df[columns])
        self._valid_df[columns], _ = DataWrapper.min_max_scale(self._valid_df[columns], train_min, train_max)
        self._test_df[columns], _ = DataWrapper.min_max_scale(self._test_df[columns], train_min, train_max)

        train_min.to_csv(os.path.join(self.output_dir, 'stats', 'min.csv'))
        train_max.to_csv(os.path.join(self.output_dir, 'stats', 'max.csv'))

    def scale_quantiles_data(self, columns: List[str]):
        """
        Scales the training, validation, and test DataFrames using the median and interquartile range (IQR) of the training data.
        
        The median, 25th percentile (q25), and 75th percentile (q75) values are calculated from the training data and then applied to the validation and test data to scale them.
        The median, q25, and q75 values are also saved as CSV files in the 'stats' directory within the output directory.
        """
        self._train_df[columns], (train_median, train_q25, train_q75) = DataWrapper.scale_quantiles(self._train_df[columns])
        self._valid_df[columns], _ = DataWrapper.scale_quantiles(self._valid_df[columns], train_median, train_q25, train_q75)
        self._test_df[columns], _ = DataWrapper.scale_quantiles(self._test_df[columns], train_median, train_q25, train_q75)

        train_median.to_csv(os.path.join(self.output_dir, 'stats', 'median.csv'))
        train_q25.to_csv(os.path.join(self.output_dir, 'stats', 'q25.csv'))
        train_q75.to_csv(os.path.join(self.output_dir, 'stats', 'q75.csv'))

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

    def fillna(self, col_name: str, method: str):
        if "zeroes" == method:
            self._df[col_name].fillna(0, inplace=True)
        elif "mean" == method:
            self._df[col_name].fillna(self._df[col_name].dropna().mean(), inplace=True)
        elif "median" == method:
            self._df[col_name].fillna(self._df[col_name].dropna().median(), inplace=True)
        elif "most_freq" == method:
            self._df[col_name].fillna(self._df[col_name].dropna().mode().values[0], inplace=True)
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