
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

## Training figures
#with open('../Results_Single_Models_Backup.pkl', 'rb') as input:
#    res = pickle.load(input)
#
#with open('../Results_Ensemble_Models_Backup.pkl', 'rb') as input:
#    eres = pickle.load(input)


# Training Loss Per Iteration
iter_train_loss = pd.concat((pd.DataFrame(res.iter_train_loss, columns=['ResNet56']), 
                             pd.DataFrame.from_dict(eres.iter_train_loss)), axis=1)

# Training Loss Per Epoch
epoch_train_loss = pd.concat((pd.DataFrame(res.train_loss, columns=['ResNet56']), 
                              pd.DataFrame.from_dict(eres.train_loss)), axis=1)

# Training Accuracy Per Iteration
iter_train_accy = pd.concat((pd.DataFrame(res.iter_train_accy, columns=['ResNet56']), 
                             pd.DataFrame.from_dict(eres.iter_train_accy)), axis=1)

# Training Accuracy Per Epoch
epoch_train_accy = pd.concat((pd.DataFrame(res.train_accy, columns=['ResNet56']), 
                              pd.DataFrame.from_dict(eres.train_accy)), axis=1)

# Training Test Error Per Iteration
iter_train_testerror = 100 - iter_train_accy.iloc[:,:]

# Training Test Error Per Epoch
epoch_train_testerror = 100 - iter_train_accy.iloc[:,:]


# Validation Loss
valid_loss = pd.concat((pd.DataFrame(res.valid_loss, columns=['ResNet56']), 
                        pd.DataFrame.from_dict(eres.valid_loss)), axis=1)

# Validation Accuracy 
valid_accy = pd.concat((pd.DataFrame(res.valid_accy, columns=['ResNet56']), 
                        pd.DataFrame.from_dict(eres.train_accy)), axis=1)

# Validation Test Error
valid_testerror = 100 - valid_accy.iloc[:,:]


# TRAINING DATA
train = {'iter': 
    
            {'loss': iter_train_loss,
             'accy': iter_train_accy,
             'test': iter_train_testerror
             },

         'epoch':
    
            {'loss': epoch_train_loss,
             'accy': epoch_train_accy,
             'test': epoch_train_testerror
             },
   }


# VALIDATION DATA
valid = {'loss': valid_loss,
         'accy': valid_accy,
         'test': valid_testerror
         }



# TESTING DATA
test = {'single': ,
        'ensemble': ,}



# DATA DICTIONARY
data = {'train': 0,
        'valid': 0,
        'test': 0}





app = dash.Dash()

options = ['Single Model', 'Ensemble Model'] + [str(k) for k in eres.train_accy.keys()]
x_axis_options = ['Iterations', 'Epochs']
dataset_options = ['Loss', 'Accuracy', 'Test Error']

app.layout = html.Div([
        
        dcc.Dropdown(id='measure-picker',
                     options = dataset_options,
                     value = dataset_options[1]),
    
        dcc.Graph(id='graph'),
        
        dcc.Dropdown(id='season-picker',
                     options = x_axis_options,
                     value = x_axis_options[1])])
    
    
@app.callback(Output(component_id = 'graph', component_property='figure'),
              [Input(component_id = 'measure-picker', component_property='value'),
               Input(component_id = 'resolution-picker', component_property='value')])    
def update_graph(selected_season):
    
    
    ## TODO: Conversor pickers to dict keys
    
    
    
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
