from IPython.display import display_html
from itertools import chain, cycle
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def display_side_by_side(*args, titles=cycle([''])):
    """Given a list of pd.DataFrame or pd.Series, display their HTMLs side by side with the titles provided"""

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
    """Given a pd.Dataframe, display its HTML table"""
    display_html(table.to_html(), raw=True)


def display_2d_hyperparameter_grid_search(hyperparameter_1: list,
                                          hyperparameter_2: list,
                                          results: dict,
                                          names: tuple[str, str] = ("x", "y"),
                                          ):
    """
    Given a dictionary of type dict[hyperparameter1][hyperparameter2] = loss,
    Given the hyperparameter values in a list
    Given the names of the x and y axis

    Plot a 3d graph showing the loss wrt to both hyperparameters
    """
    data = np.zeros((len(hyperparameter_1), len(hyperparameter_2)))

    for i, hy1 in enumerate(hyperparameter_1):
        for j, hy2 in enumerate(hyperparameter_2):
            data[i, j] = results[(hy1, hy2)]

    z_data = pd.DataFrame(data)

    fig = go.Figure(data=[
        go.Surface(x=list(hyperparameter_2), y=list(hyperparameter_1), z=z_data)
    ])
    fig.update_traces(contours_z=dict(show=True,
                                      usecolormap=True,
                                      highlightcolor="limegreen",
                                      project_z=True))
    fig.update_layout(title=dict(
            text='Hyperparameter tuning graph'),
            autosize=False,
            scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
            width=500, height=500,
            margin=dict(l=65, r=50, b=65, t=90),
            scene=dict(
                    xaxis_title=names[1],
                    yaxis_title=names[0],
                    zaxis_title="Accuracy"
            ),
            template="plotly_dark"
    )

    fig.show()
