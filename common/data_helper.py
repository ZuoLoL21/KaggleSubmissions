import matplotlib.pyplot as plt
import seaborn as sns

from common.display_helper import *


def summarize_categorical_columns_wrt_target(
		df: pd.DataFrame,
		variable_cols: list | str,
		target_col: list | str,
		agg_funcs: list | tuple | dict = ('mean', 'sum', 'count'),
):
	display_table(
			pd.pivot_table(df, index=variable_cols, values=target_col, aggfunc=agg_funcs).reset_index()
	)


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


def summarize_continuous_columns_wrt_target(df: pd.DataFrame, variable_cols: str, col: str, row: str = None):
	# Plot survival per Age bin
	if row is None:
		g = sns.FacetGrid(df, col=col, row=row)
	else:
		g = sns.FacetGrid(df, col=col, row=row, height=2, aspect=1.6)

	g.map(plt.hist, variable_cols, bins=20)
	plt.show()


def show_distribution(df: pd.DataFrame,
					  variable_col: str,
					  target_col: str,
					  possible_vals: tuple = (0, 1),
					  colors: tuple = ("green", "red"),
					  max_x: int = None,
					  min_x: int = 0,
					  ):
	for val, color in zip(possible_vals, colors):
		sns.kdeplot(df[variable_col][df[target_col] == val], color=color)

	plt.legend([f"{target_col} = {x}" for x in possible_vals])
	plt.xlim(min_x, max_x)
	plt.show()
