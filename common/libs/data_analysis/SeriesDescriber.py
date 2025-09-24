import matplotlib.pyplot as plt
import seaborn as sns

from common.libs.DataDisplayer import *


def describe_categorical_series(series: pd.Series):
    display_side_by_side(
            series.astype('category').describe(),
            series.value_counts(dropna=False),
            titles=["Description", "Count"])


def describe_continuous_series(series: pd.Series):
    display_side_by_side(
            series.describe(),
            series.value_counts(dropna=False, bins=10),
            titles=["Description", "Count"])


def show_distribution(series: pd.Series, min_x: int = 0, max_x: int = None):
    sns.kdeplot(series, color="green")
    plt.xlim(min_x, max_x)
    plt.show()
