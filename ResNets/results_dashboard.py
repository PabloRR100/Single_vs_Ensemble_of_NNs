
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


# Helper functions

def roundup(x):
    x = x.max()
    return int(math.ceil(x / 10.0)) * 10

def rounddown(x):
    x = x.min()
    return int(math.floor(x / 10.0)) * 10

def drawline(c):
    
    if c in ['ensemble', 'ResNet56']:    
        line = dict(width = 2, color = 'red' if c == 'ensemble' else 'blue')
    else:
        line = dict(width = 0.8)
        
    return line


# Data Wrapper

# Training figures
with open('Results_Single_Models.pkl', 'rb') as input:
    res = pickle.load(input)

with open('Results_Ensemble_Models.pkl', 'rb') as input:
    eres = pickle.load(input)

## Training figures
#with open('../Results_Single_Models_Backup.pkl', 'rb') as input:
#    res = pickle.load(input)
#
#with open('../Results_Ensemble_Models_Backup.pkl', 'rb') as input:
#    eres = pickle.load(input)

data = aggregateResults(res, eres)

#options = ['Single Model', 'Ensemble Model'] + [str(k) for k in eres.train_accy.keys()]
resolutions = [{'label': 'Iteration', 'value': 'iter'},
               {'label': 'Epochs', 'value': 'epoch'}]

measurements = [{'label': 'Loss', 'value': 'loss'},
                {'label': 'Accuracy', 'value': 'accy'},
                {'label': 'Test Error', 'value': 'test'}]



# Dashboard Layout

app = dash.Dash()
app.layout = html.Div([

    html.Div([
        html.H1('ResNet56 vs ResNet20 (x3)')
    ], style = {'text-align':'center'}),
        
    html.Div([
        
        # Training results
        html.Div([
            
            dcc.Dropdown(id='measure-picker',
                 options = measurements,
                 value = 'test'),
    
            dcc.Graph(id='train_graph'),
            
            dcc.Dropdown(id='resolution-picker',
                         options = resolutions,
                         value = 'epoch')
                
        ], className = 'six columns'),
                
        # Validation results
        html.Div([
                
            dcc.Dropdown(id='valid-measure-picker',
                 options = measurements,
                 value = 'test'),
    
            dcc.Graph(id='valid-graph'),
                
        ], className = 'six columns')
            
    ], className = 'row'),
    
    
    
    html.Div([
        
        # Time Analysis
        html.Div([                
            dcc.Graph(id='test_graph'),                
        ], className = 'eigth columns'),
                
#        # Test results
#        html.Div([
#            dcc.Graph(id='test-graph'),                
#        ], className = 'four columns')
#            
    ], className = 'row'),
            
])

# Styling
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})
    

@app.callback(Output(component_id = 'train_graph', component_property='figure'),
              [Input(component_id = 'measure-picker', component_property='value'),
               Input(component_id = 'resolution-picker', component_property='value')])    
def train_graph(measure, resolution):
    
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
                        line = drawline(col),
                        visible = True if col in ['ensemble', 'ResNet56'] else 'legendonly'
                        )
                )
        
    layout = go.Layout(title='Training Results',
                       xaxis = {'title': resolution},
                       yaxis = {'title': measure, 
                                'range': [rounddown(df[col]), roundup(df[col])],
                                'dtick': 10,},
                       hovermode = 'closest')
    
    return {'data': traces, 'layout': layout}


@app.callback(Output(component_id = 'valid-graph', component_property='figure'),
              [Input(component_id = 'valid-measure-picker', component_property='value')])    
def valid_graph(measure):
    
    df = data['valid'][measure]
    
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
                        line = drawline(col),
                        visible = True if col in ['ensemble', 'ResNet56'] else 'legendonly'
                        )
                )
        
    layout = go.Layout(title='Validation Data',
                       xaxis = {'title': 'Epochs'},
                       yaxis = {'title': measure, 
                                'range': [rounddown(df[col]), roundup(df[col])],
                                'dtick': 10,},
                       hovermode = 'closest')
    
    return {'data': traces, 'layout': layout}

    
if __name__ == '__main__':
    app.run_server(debug=True)
