
import os
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objs as go

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from results import aggregateResults


## Training figures
#with open('Results_Single_Models.pkl', 'rb') as input:
#    res = pickle.load(input)
#
#with open('Results_Ensemble_Models.pkl', 'rb') as input:
#    eres = pickle.load(input)

# Training figures
with open('../Results_Single_Models_Backup.pkl', 'rb') as input:
    res = pickle.load(input)

with open('../Results_Ensemble_Models_Backup.pkl', 'rb') as input:
    eres = pickle.load(input)

data = aggregateResults(res, eres)


app = dash.Dash()

#options = ['Single Model', 'Ensemble Model'] + [str(k) for k in eres.train_accy.keys()]
resolutions = [{'label': 'Iteration', 'value': 'iter'},
               {'label': 'Epochs', 'value': 'epoch'}]

measurements = [{'label': 'Loss', 'value': 'loss'},
                {'label': 'Accuracy', 'value': 'accy'},
                {'label': 'Test Error', 'value': 'accy'}]


app.layout = html.Div([
        
        # Training results
        html.Div([

                html.H1('Training Results'),
                
                dcc.Dropdown(id='measure-picker',
                     options = measurements,
                     value = measurements[1]),
    
                dcc.Graph(id='graph'),
                
                dcc.Dropdown(id='resolution-picker',
                             options = resolutions,
                             value = resolutions[1])
                ])
        ])


@app.callback(Output(component_id = 'graph', component_property='figure'),
              [Input(component_id = 'measure-picker', component_property='value'),
               Input(component_id = 'resolution-picker', component_property='value')])    
def train_graph(measure, resolution):
        
    mlab = measure['label']
    mval = measure['value']
    
    rlab = resolution['laeb']
    rval = resolution['value']
    
    # Return data only for the selection from the dropdown
    df = data['train'][rval][mval]

    
    traces = []
    
    for col in df:
        
        traces.append(
                go.Scatter(
                        #x = np.range(len(list(df))), 
                        x = df.index.values+1, 
                        y = df[col],
                        name = col,
                        mode = 'lines+markers',
                        marker = {'size': 5},
                        visible = True if col in ['ensemble', 'ResNet56'] else 'legendonly'
                        )
                )
        
    layout = go.Layout(title='Training Results',
                       xaxis = {'title': rlab},
                       yaxis = {'title': mlab, 
                                'range': [0, 100],
                                'dtick': 10,},
                       hovermode = 'closest')
    
    return {'data': traces, 'layout': layout}
    
if __name__ == '__main__':
    app.run_server(debug=True)
