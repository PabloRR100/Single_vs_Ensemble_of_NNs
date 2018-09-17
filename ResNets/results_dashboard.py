
import os
import math
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objs as go

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from results import aggregateResults


# Training figures
with open('Results_Single_Models.pkl', 'rb') as input:
    res = pickle.load(input)

with open('Results_Ensemble_Models.pkl', 'rb') as input:
    eres = pickle.load(input)
#
## Training figures
#with open('../Results_Single_Models_Backup.pkl', 'rb') as input:
#    res = pickle.load(input)
#
#with open('../Results_Ensemble_Models_Backup.pkl', 'rb') as input:
#    eres = pickle.load(input)

data = aggregateResults(res, eres)


app = dash.Dash()

#options = ['Single Model', 'Ensemble Model'] + [str(k) for k in eres.train_accy.keys()]
resolutions = [{'label': 'Iteration', 'value': 'iter'},
               {'label': 'Epochs', 'value': 'epoch'}]

measurements = [{'label': 'Loss', 'value': 'loss'},
                {'label': 'Accuracy', 'value': 'accy'},
                {'label': 'Test Error', 'value': 'test'}]


app.layout = html.Div([
        
        # Training results
        html.Div([

                html.H1('Training Results'),
                
                dcc.Dropdown(id='measure-picker',
                     options = measurements,
                     value = 'test'),
    
                dcc.Graph(id='graph'),
                
                dcc.Dropdown(id='resolution-picker',
                             options = resolutions,
                             value = 'epoch')
                ])
        ])


@app.callback(Output(component_id = 'graph', component_property='figure'),
              [Input(component_id = 'measure-picker', component_property='value'),
               Input(component_id = 'resolution-picker', component_property='value')])    
def train_graph(measure, resolution):
        
    print(measure)
    print(resolution)
    
    def roundup(x):
        x = x.max()
        return int(math.ceil(x / 10.0)) * 10
    
    def rounddown(x):
        x = x.min()
        return int(math.floor(x / 10.0)) * 10
    
    def setcolor(x):
        colors = {'ensemble': 'red',
                  'ResNet56': 'blue'}
        return colors[x]
    
#    mlab = measure['label']
#    rlab = resolution['label']
#    
#    mval = measure['value']
#    rval = resolution['value']
    
    # Return data only for the selection from the dropdown
#    df = data['train'][ ][mval]
    
    df = data['train'][resolution][measure]

    
    traces = []
    
    for col in df:
        
        traces.append(
                go.Scatter(
                        #x = np.range(len(list(df))), 
                        x = df.index.values+1, 
                        y = df[col],
                        name = col,
                        mode = 'lines+markers' if col in ['ensemble', 'ResNet56'] else 'lines',
                        marker = {'size': 5},
                        line = dict(
                                width = 2 if col in ['ensemble', 'ResNet56'] else 0.8,
                                dash = 'dash' if col not in ['ensemble', 'ResNet56'] else 'solid'),
                        visible = True if col in ['ensemble', 'ResNet56'] else 'legendonly'
                        )
                )
        
    layout = go.Layout(title='Training Results',
                       xaxis = {'title': 'Testing'},
                       yaxis = {'title': 'Testing', 
                                'range': [rounddown(df[col]), roundup(df[col])],
                                'dtick': 10,},
                       hovermode = 'closest')
    
    return {'data': traces, 'layout': layout}
    
if __name__ == '__main__':
    app.run_server(debug=True)
