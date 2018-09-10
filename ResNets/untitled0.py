

import gym
env = gym.make('FrozenLake-v0')

import os
root = '/Users/pabloruizruiz/OneDrive/Proyectos/Betcomm'
os.chdir(root)

import sys
sys.path.append(root)

if os.path.exists(root):
    data_folder = os.path.join(root, 'Datos')
    path_to_data = os.path.join(data_folder, 'Scrapped')
    path_to_save = os.path.join(data_folder, 'Created')

import sys
import django
sys.path.append(os.path.join(root, 'Betcomm'))
os.environ['DJANGO_SETTINGS_MODULE'] = 'betcomm.settings'
django.setup()

import numpy as np
import pandas as pd
#import plotly.offline as pyo
import plotly.graph_objs as go

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


# Bring Classifications
from SQL.queries import conn, classification_query
#from seasonalModels.models import Classification
class_global = pd.read_sql_query(classification_query, conn)

app = dash.Dash()

temporadas = []
for temp in class_global['season'].unique():
    temporadas.append({'label': str(temp), 'value': temp})


app.layout = html.Div([
        
        dcc.Graph(id='graph'),
        dcc.Dropdown(id='season-picker',
                     options = temporadas,
                     value = class_global['season'].max())]) # Most recent (current)
    

@app.callback(Output(component_id = 'graph', component_property='figure'),
              [Input(component_id = 'season-picker', component_property='value')])    
def update_figure(selected_season):
    # Return data only for the selection from the dropdown
    filtered_df = class_global[class_global['season'] == selected_season]
    
    traces = []
    
    for team in filtered_df['team_name'].unique():
        df_by_team = filtered_df[filtered_df['team_name'] == team]
        traces.append(go.Scatter(x = df_by_team['round'], 
                                 y = df_by_team['position'],
                                 name = team,
                                 mode = 'lines+markers',
                                 hoverinfo = 'name' + '. ' + 'Jornada: ' + 'x' + ', Posición: ' + 'y',
                                 #opacity = 0.7,
                                 marker = {'size': 5},
                                 visible = True if team in ['Real Madrid', 'Barcelona'] else 'legendonly'))
        
    layout = go.Layout(title='Clasificación',
                       xaxis = {'title': 'Jornadas'},
                       yaxis = {'title': 'Posicion', 
                                'autorange': 'reversed',
                                'range': [0, 20],
                                'dtick': 1,},
                       hovermode = 'closest')
    
    return {'data': traces, 'layout': layout}
    
if __name__ == '__main__':
    app.run_server()

