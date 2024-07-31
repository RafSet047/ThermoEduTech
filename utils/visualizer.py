import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import *

class Visualizer:
    def __init__(self, df):
        """
        Initialize the Visualizer with a pandas DataFrame.
        
        Args:
        df (pandas.DataFrame): The data to be visualized.
        """
        self._df = df
        self._figsize = (10, 6)
        sns.set_theme(style="whitegrid")  # Set a seaborn style for all plots

    @property
    def df(self):
        return self._df

    def set_figsize(self, new_figsize: Tuple[int, int]):
        """
        Setter for the figsize

        Args:
            new_figsize (Tuple[int, int]): new size of the plots
        """
        self._figsize = new_figsize
    
    def _prettify_plot(self, title, xlabel, ylabel):
        """
        Helper method to prettify the plot by adding titles and adjusting layout.
        
        Args:
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        """
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.xticks(rotation=45)
        plt.tight_layout()
    
    def lineplot(self, x, y, **kwargs):
        """
        Create a line plot.
        
        Args:
        x (str): The column name for the x-axis.
        y (str): The column name for the y-axis.
        kwargs: Additional keyword arguments to pass to seaborn.lineplot.
        """
        plt.figure(figsize=self._figsize)
        sns.lineplot(data=self._df, x=x, y=y, **kwargs)
        self._prettify_plot(f'Line Plot of {y} vs {x}', x, y)
        plt.show()
    
    def histplot(self, column, **kwargs):
        """
        Create a histogram.
        
        Args:
        column (str): The column name to plot.
        kwargs: Additional keyword arguments to pass to seaborn.histplot.
        """
        plt.figure(figsize=self._figsize)
        sns.histplot(data=self._df, x=column, kde=True, **kwargs)
        self._prettify_plot(f'Histogram of {column}', column, 'Frequency')
        plt.show()
    
    def scatterplot(self, x, y, sample_size=1., **kwargs):
        """
        Create a scatter plot.
        
        Args:
        x (str): The column name for the x-axis.
        y (str): The column name for the y-axis.
        kwargs: Additional keyword arguments to pass to seaborn.scatterplot.
        """
        plt.figure(figsize=self._figsize)
        sns.scatterplot(data=self._df.sample(frac=sample_size), x=x, y=y, **kwargs)
        self._prettify_plot(f'Scatter Plot of {y} vs {x}', x, y)
        plt.show()
    
    def boxplot(self, x, y, **kwargs):
        """
        Create a box plot.
        
        Args:
        x (str): The column name for the x-axis (can be categorical).
        y (str): The column name for the y-axis.
        kwargs: Additional keyword arguments to pass to seaborn.boxplot.
        """
        plt.figure(figsize=self._figsize)
        sns.boxplot(data=self._df, x=x, y=y, **kwargs)
        self._prettify_plot(f'Box Plot of {y} by {x}', x, y)
        plt.show()
    
    def violinplot(self, x, y, **kwargs):
        """
        Create a violin plot.
        
        Args:
        x (str): The column name for the x-axis (can be categorical).
        y (str): The column name for the y-axis.
        kwargs: Additional keyword arguments to pass to seaborn.violinplot.
        """
        plt.figure(figsize=self._figsize)
        sns.violinplot(data=self._df, x=x, y=y, **kwargs)
        self._prettify_plot(f'Violin Plot of {y} by {x}', x, y)
        plt.show()
    
    def swarmplot(self, x, y, **kwargs):
        """
        Create a swarm plot.
        
        Args:
        x (str): The column name for the x-axis (can be categorical).
        y (str): The column name for the y-axis.
        kwargs: Additional keyword arguments to pass to seaborn.swarmplot.
        """
        plt.figure(figsize=self._figsize)
        sns.swarmplot(data=self._df, x=x, y=y, **kwargs)
        self._prettify_plot(f'Swarm Plot of {y} by {x}', x, y)
        plt.show()
    
    def heatmap(self, columns: Optional[List[str]] = None, show_target_only: bool = False, **kwargs):
        """
        Create a heatmap of the correlation matrix.
        Args:
        columns (list[str]|None): list of specified columns in order to calculate the correlation, if None will be done on all columns
        show_target_only (bool): whether to show only the target column's row as heatmap (NOTE target column is the last from `columns`)
        kwargs: Additional keyword arguments to pass to seaborn.swarmplot.
        
        """
        plt.figure(figsize=self._figsize)
        data = None
        if columns is None:
            columns = list(self._df.columns)
            data = self._df.corr()
        else:
            data = self._df[columns].corr()

        if show_target_only:
            data = data[[columns[-1]]]

        sns.heatmap(data, annot=True, vmin=-1., vmax=1., cmap='coolwarm', **kwargs)
        self._prettify_plot('Correlation Heatmap', '', '')
        plt.show()

    @staticmethod
    def plot_train_curves(train_losses: List[float], valid_losses: List[str], save_path: Optional[str] = None) -> None:
        """
        Plots the training and validation losses over epochs using Seaborn.

        Args:
            train_losses (list of float): List of training losses for each epoch.
            val_losses (list of float): List of validation losses for each epoch.
        """
        epochs = range(1, len(train_losses) + 1)
        data = pd.DataFrame({
            'epoch': epochs,
            'train_loss': train_losses,
            'valid_loss': valid_losses
        })
        data = data.melt(id_vars='epoch', var_name='Loss Type', value_name='Loss')
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=data, x='epoch', y='Loss', hue='Loss Type', marker='o')
        
        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.tight_layout()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)

    @staticmethod
    def plot_predictions(y_true: List[float], y_pred: List[str], save_path: Optional[str] = None) -> None:
        """
        Plots the training and validation losses over epochs using Seaborn.

        Args:
            train_losses (list of float): List of training losses for each epoch.
            val_losses (list of float): List of validation losses for each epoch.
        """
        hours = range(1, len(y_true) + 1)
        data = pd.DataFrame({
            'hours': hours,
            'Actual': y_true,
            'Predictions': y_pred
        })
        data = data.melt(id_vars='hours', var_name='values', value_name='true_vs_pred')
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=data, x='hours', y='true_vs_pred', hue='values', marker='o')
        
        plt.title('Test set Actual and Predicted values')
        plt.xlabel('Hours')
        plt.ylabel('Temperature')
        plt.grid(True)
        plt.tight_layout()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)

if __name__ == "__main__":
    tr = [0.9, 0.87, 0.79, 0.54, 0.36, 0.22]
    vl = [1.5, 1.17, 0.99, 0.62, 0.75, 0.82]
    Visualizer.plot_train_curves(tr, vl, 'tmp.png')