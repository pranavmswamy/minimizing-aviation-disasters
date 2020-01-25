# -*- coding: utf-8 -*-
"""
Created on Sat May 25 17:24:10 2019

@author: Pranav
"""

import pandas as pd
import time
import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objs as go
from collections import deque

train_df = pd.read_csv('train_onepilot.csv')

train_df = train_df.sort_values(by='time').sort_index()

ecglist = list(train_df['ecg'])
gsrlist = list(train_df['gsr'])
rlist = list(train_df['r'])
eventlist = list(train_df['event'])


app = dash.Dash('pilot-alert-system')
max_length = 300;
times = deque(maxlen = max_length)
ecgs = deque(maxlen = max_length)
rs = deque(maxlen = max_length)
gsrs = deque(maxlen = max_length)
events = deque(maxlen = max_length)

data_dict = {"ECG":ecgs,
"RESP": rs,
"GSR": gsrs}

def update_values(times, ecgs, rs, gsrs, events):
    times.append(time.time())
    predictedvalue = "00"
    if len(times) == 1:
        ecgs.append(ecglist[0])
        rs.append(rlist[0])
        gsrs.append(gsrlist[0])
        events.append(eventlist[0])
    else:
        ecgs.append(ecglist[len(times)]) #loops for only the first max_len values
        rs.append(rlist[len(times)])
        gsrs.append(gsrlist[len(times)])
        events.append(eventlist[len(times)])
        predictedvalue = eventlist[len(times)]
        
    return times, ecgs, rs, gsrs, events

times, ecgs, rs, gsrs, events = update_values(times, ecgs, rs, gsrs, events)

app.layout = html.Div([
    html.Div([
        html.H3('--PILOT ALERT SYSTEM--',
                style={'float': 'center',
                       'textAlign': 'center',
                       'fontFamily': 'Times New Roman',
                       'fontStyle': 'Bold'
                       }),
        ]),
    html.Div(children=html.Div(id='graphs'), className='row',style = {"border" : "2px black solid",'margin-left':20,'margin-right':20, 'padding': '10px', 'backgroundColor': '#D4DFE1'}),
    dcc.Interval(
        id='graph-update',
        interval=1000,
        n_intervals=0),
    html.Div([html.H5('FOCUSED', id='predictedvalue',
                style={'float': 'center',
                       'textAlign': 'center',
                       'backgroundColor': '#55FC7A', 'margin-left':70,'margin-right':70,
                       'fontFamily': 'Times New Roman',
                       }),
                html.H5('DISTRACTED', id='predictedvalue1',
                style={'float': 'center',
                       'textAlign': 'center',
                       'fontFamily': 'Times New Roman',
                       }),
                html.H5('STARTLED', id='predictedvalue2',
                style={'float': 'center',
                       'textAlign': 'center',
                       'fontFamily': 'Times New Roman',
                       })], className = 'row'), 
    ], className="container",style={'width':'98%','margin-left':20,'margin-right':20, 'margin-top': 12, 'margin-bottom' : 12,'max-width':5000, "border":"2px black solid", 'backgroundColor':'#EDF6F8'})
        
@app.callback(
    Output('graphs', 'children'),
    [Input('graph-update','n_intervals')]
    )

def update_graph(n):
    graphs = []
    update_values(times, ecgs, rs, gsrs, events) 

    for data_name in ["ECG","RESP","GSR"]:

        data = go.Scatter(
            x=list(times),
            y=list(data_dict[data_name]),
            name='Scatter',
            fill="tozeroy",
            fillcolor="#6CD0E5"
            )

        graphs.append(html.Div(dcc.Graph(
            id=data_name,
            animate=True,
            figure={'data': [data],'layout' : go.Layout(xaxis=dict(range=[min(times),max(times)]),
                                                        yaxis=dict(range=[min(data_dict[data_name]),max(data_dict[data_name])]),
                                                        margin={'l':50,'r':50,'t':50,'b':1},
                                                        title='{}'.format(data_name + " (microvolts)"))}
            ), className='col s12 m6 l4'))
    
    
    
    return graphs



external_css = ["https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css"]
for css in external_css:
    app.css.append_css({"external_url": css})

external_js = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/js/materialize.min.js']
for js in external_js:
    app.scripts.append_script({'external_url': js})


if __name__ == '__main__':
    app.run_server(debug=False)