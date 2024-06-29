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
    
    def scatterplot(self, x, y, **kwargs):
        """
        Create a scatter plot.
        
        Args:
        x (str): The column name for the x-axis.
        y (str): The column name for the y-axis.
        kwargs: Additional keyword arguments to pass to seaborn.scatterplot.
        """
        plt.figure(figsize=self._figsize)
        sns.scatterplot(data=self._df, x=x, y=y, **kwargs)
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
    
    def heatmap(self, columns: Optional[List[str]] = None, **kwargs):
        """
        Create a heatmap of the correlation matrix.
        Args:
        columns (list[str]|None): list of specified columns in order to calculate the correlation, if None will be done on all columns
        kwargs: Additional keyword arguments to pass to seaborn.swarmplot.
        
        """
        plt.figure(figsize=self._figsize)
        sns.heatmap(self._df.corr(), annot=True, cmap='coolwarm', **kwargs)
        self._prettify_plot('Correlation Heatmap', '', '')
        plt.show()
