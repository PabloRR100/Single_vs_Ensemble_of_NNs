 
import math
import pickle
import plotly.graph_objs as go

import dash
import dash_auth
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from results import aggregateResults

USERNAME_PASSWORD_PAIRS = [
        ['PabloRR10', 'Nartenalpo10-'],
        ['wasay', 'wasay'],
        ['username', 'password']]


import os
import sys
sys.path.append('..')
results_path = os.path.abspath('../results/dicts/ResNet110')

# Helper functions

def roundup(x):
    return int(math.ceil(x.max() / 10.0)) * 10

def rounddown(x):
    return int(math.floor(x.min() / 10.0)) * 10

def drawline(c):
    
    if c in ['ensemble', 'ResNet110']:    
        line = dict(width = 2, color = 'red' if c == 'ensemble' else 'blue')
    else:
        line = dict(width = 0.8)
        
    return line


# Data Wrapper

# Training figures
with open(os.path.join(results_path, 'Results_Single_Models.pkl'), 'rb') as input:
    res = pickle.load(input)

with open(os.path.join(results_path, 'Results_Ensemble_Models.pkl'), 'rb') as input:
    eres = pickle.load(input)
    
    
with open(os.path.join(results_path, 'Results_Testing.pkl'), 'rb') as input:
    test = pickle.load(input)


data = aggregateResults(res, eres, test)

resolutions = [{'label': 'Iteration', 'value': 'iter'},
               {'label': 'Epochs', 'value': 'epoch'}]

measurements = [{'label': 'Loss', 'value': 'loss'},
                {'label': 'Accuracy', 'value': 'accy'},
                {'label': 'Test Error', 'value': 'test'}]

reversed_dictionary = {'loss': 'Loss', 
                       'accy':'Accuracy',
                       'test':'Test Error'}


def time_graph():
    
    # Line plot in backup code at the end of the file
    d = data['timer']
    c = ['rgba(55, 128, 191, 0.7)', 'rgba(219, 64, 82, 0.7)']
    
    x = list(d)
    y = d.values[-1,:]
    
    traces = [go.Bar(x = x, y = y, 
                     marker = {'color':c})]
                
    layout = go.Layout(title='Training time',
                       xaxis={'title':'Epochs'},
                       yaxis={'title':'Minutes'},)
    
    annotations = []
    for i in range(0, 2):
        annotations.append(dict(x=x[i], y=y[i], text=y[i],
                                yanchor = 'top',
                                font=dict(family='Arial', size=30,
                                color='rgba(245, 246, 249, 1)'),
                                showarrow=False,))
        layout['annotations'] = annotations

    return {'data': traces, 'layout':layout}


def test_graph():
    
    d = data['test']
    c = ['rgba(55, 128, 191, 0.7)', 'rgba(219, 64, 82, 0.7)']
    
    x = ['Deep Model', 'Ensemble Model']
    y = [d['single'], d['ensemble']]
    
    traces = [go.Bar(x = x, y = y, 
                     marker = {'color':c})]
        
    layout = go.Layout(title='Testing Set',
                  xaxis={'title':'Models'},
                  yaxis={'title':'Accuracy (%)'},)

    annotations = []
    for i in range(0, 2):
        annotations.append(dict(x=x[i], y=y[i], text=y[i],
                                yanchor = 'top',
                                font=dict(family='Arial', size=30,
                                color='rgba(245, 246, 249, 1)'),
                                showarrow=False,))
        layout['annotations'] = annotations

    return {'data': traces, 'layout':layout}
        
         

# Dashboard Layout

app = dash.Dash()
server = app.server
auth = dash_auth.BasicAuth(app, USERNAME_PASSWORD_PAIRS)

app.layout = html.Div([

    html.Div([
        html.H1('ResNet110 vs ResNet20 (x6)')
    ], id = 'title', style = {'text-align':'center'}),
        
    html.Div([
        
        # Training results
        html.Div([       
                
            # Dropdowns
            html.Div([            
                    
                # Metric Dropdown
                html.Div([                    
                    dcc.Dropdown(
                        id='measure-picker',
                        options = measurements,
                        value = 'test'),
                ], className = 'six columns'),                
                            
                # Metric Dropdown
                html.Div([
                    dcc.Dropdown(
                        id='resolution-picker',
                        options = resolutions,
                        value = 'epoch')
                ], className = 'six columns')
                
            ],  className = 'row'),
                
            # Graph
            html.Div([
                dcc.Graph(id='train_graph')
            ])
                    
        ], className = 'six columns'),
                
        # Validation results
        html.Div([
            
            # Dropdown
            html.Div([            
                    
                # Metric Dropdown
                html.Div([       
                    dcc.Dropdown(
                        id='valid-measure-picker',
                        options = measurements,
                        value = 'test')
                ], className = 'six columns'),
                    
                # Metric Dropdown
                html.Div([], className = 'six columns')
                
            ], className = 'row'), 
                
            # Graph
            html.Div([
                dcc.Graph(id='valid-graph'),
            ])
                    
        ], className = 'six columns')
            
    ], className = 'row'),
    
        
    html.Div([
        
        # Time Analysis
        html.Div([                
             
            dcc.Graph(
                id='time-graph',
                figure=time_graph()
            )
            
        ], className = 'six columns'),
                
        # Test results
        html.Div([
            dcc.Graph(
                id='test-graph',
                figure=test_graph()
            )                
        ], className = 'six columns')
            
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
                        mode = 'lines+markers' if col in ['ensemble', 'ResNet110'] else 'lines',
                        marker = {'size': 5},
                        line = drawline(col),
                        visible = True if col in ['ensemble', 'ResNet110'] else 'legendonly'
                        )
                )
                
    layout = go.Layout(title='Training Set',
                       xaxis = {'title': resolution},
                       yaxis = {'title': reversed_dictionary[measure], 
                                'range': [rounddown(df.min()), roundup(df.max())],
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
                        mode = 'lines+markers' if col in ['ensemble', 'ResNet110'] else 'lines',
                        marker = {'size': 5},
                        line = drawline(col),
                        visible = True if col in ['ensemble', 'ResNet110'] else 'legendonly'
                        )
                )
        
    layout = go.Layout(title='Validation Set',
                       xaxis = {'title': 'Epochs'},
                       yaxis = {'title': reversed_dictionary[measure], 
                                'range': [rounddown(df.min()), roundup(df.max())],
                                'dtick': 10,},
                       hovermode = 'closest')
    
    return {'data': traces, 'layout': layout}    


if __name__ == '__main__':
    app.run_server(debug=True)
