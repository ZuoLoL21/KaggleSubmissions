import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import regex as re


def one_hot_encode(df: pd.DataFrame, categorical_columns: list | str) -> pd.DataFrame:
	if isinstance(categorical_columns, str):
		categorical_columns = [categorical_columns]

	encoder = OneHotEncoder(sparse_output=False)

	one_hot_encoded = encoder.fit_transform(df[categorical_columns])
	one_hot_df = pd.DataFrame(
			one_hot_encoded,
			columns=encoder.get_feature_names_out(categorical_columns),
			index=df.index
	)
	df.drop(categorical_columns, axis=1, inplace=True)
	df_encoded = pd.concat([df, one_hot_df], axis=1)

	return df_encoded


def apply_regex(series: pd.Series, regex):
	p = re.compile(regex)

	def matcher(x):
		if pd.isna(x):
			return None
		match = p.search(str(x))  # Convert to string
		return match.group(1) if match else None

	return series.apply(matcher)
