
import os
import pickle
import pandas as pd
import plotly.graph_objs as go

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Training figures
with open('Results_Single_Models.pkl', 'rb') as input:
    res = pickle.load(input)

with open('Results_Ensemble_Models.pkl', 'rb') as input:
    eres = pickle.load(input)


data1 = {'single':res.iter_train_accy, 
        'ensemble': eres.iter_train_accy['ensemble']}

data2 = {'single':res.iter_train_loss, 
        'ensemble': eres.iter_train_loss['ensemble']}

data3 = {'single':res.train_accy, 
        'ensemble': eres.train_accy['ensemble']}

data4 = {'single':res.train_loss, 
        'ensemble': eres.train_loss['ensemble']}

data5 = {'single':res.valid_accy, 
        'ensemble': eres.valid_accy['ensemble']}

data6 = {'single':res.valid_loss, 
        'ensemble': eres.valid_loss['ensemble']}
 

sin_iter_train_accy = pd.DataFrame.from_dict(res.iter_train_accy)
ens_iter_train_accy = pd.DataFrame.from_dict(eres.iter_train_accy)
iter_train_accy = pd.concat((sin_iter_train_accy, ens_iter_train_accy), axis=1)


iterationdata = pd.DataFrame.from_dict({'single': res.iter_train_accy, 
                                        'ensemble': eres.iter_train_accy})


app = dash.Dash()

options = ['Single Model', 'Ensemble Model'] + [str(k) for k in eres.train_accy.keys()]
x_axis_options = ['Iterations', 'Epochs']


app.layout = html.Div([
        
        dcc.Graph(id='graph'),
        dcc.Dropdown(id='season-picker',
                     options = x_axis_options,
                     value = x_axis_options[1])])
    
    
@app.callback(Output(component_id = 'graph', component_property='figure'),
              [Input(component_id = 'resolution-picker', component_property='value')])    
def update_graph(selected_season):
    # Return data only for the selection from the dropdown
    filtered_df = class_global[class_global['season'] == selected_season]
    
    traces = []
    
    
    
    for team in filtered_df['team_name'].unique():
        df_by_team = filtered_df[filtered_df['team_name'] == team]
        traces.append(go.Scatter(x = df_by_team['round'], 
                                 y = df_by_team['position'],
                                 name = team,
                                 mode = 'lines+markers',
                                 #hoverinfo = ['name' + '. ' + 'Jornada: ' + 'x' + ', Posición: ' + 'y'],
                                 opacity = 0.7,
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
