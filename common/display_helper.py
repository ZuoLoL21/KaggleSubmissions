from IPython.display import display_html
from itertools import chain, cycle
import pandas as pd


def display_side_by_side(*args, titles=cycle([''])):
	html_str = '<table><tr>'
	for df, title in zip(args, chain(titles, cycle(['']))):
		if not isinstance(df, pd.DataFrame):
			df = pd.DataFrame(df)
		html_str += '<td style="vertical-align:top; text-align:center; padding:10px">'
		html_str += f'<h3 style="text-align:center">{title}</h3>'
		html_str += df.to_html().replace('table', 'table style="display:inline"')
		html_str += '</td>'
	html_str += '</tr></table>'
	display_html(html_str, raw=True)


def display_table(table):
	display_html(table.to_html(), raw=True)
