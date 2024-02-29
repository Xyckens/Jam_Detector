from flask import Flask, render_template
from dash import Dash, dcc, html, Output, Input, State, callback
import dash_bootstrap_components as dbc
import time
import threading
import dash
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import date, timedelta
from csv_file_reader import time_diff, number_detections, avg_detections, time_stopped

# https://www.kaggle.com/datasets/tsarina/mexico-city-airbnb?select=listings1.csv
df = pd.read_csv("https://raw.githubusercontent.com/Coding-with-Adam/Dash-by-Plotly/master/Other/Monterrey/airbnb.csv")
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

cams = ['0', 'phone.mp4','172.17.4.131', '51_MF.mp4']

childs = [dbc.DropdownMenuItem("Cameras", header=True)]
i = 1
for cam in cams:
    childs.append(dbc.DropdownMenuItem(f"Cam {i}", href=f"/cam{i}"))
    i += 1

navbar = dbc.Navbar(
            dbc.Container([
                    dbc.Row([
                        dbc.Col([
                            html.Img(src=dash.get_asset_url("BA_Logo.png"), height="40px", alt="logo"),
                            dbc.NavbarBrand("Jam Detection AI Hub", className="ms-2")
                        ],
                        width={"size":"auto"})
                    ],
                    align="center", className="g-0"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Nav([
                                dbc.NavItem(dbc.NavLink("Home", href="/")),
                                dbc.NavItem(dbc.DropdownMenu(
                                        children=childs,
                                        nav=True,
                                        in_navbar=True,
                                        label="Camera Selection",
                                ))
                            ],
                            navbar=True)
                        ],
                        width={"size":"auto"})
                    ],
                    align="center"),
                    dbc.Col(dbc.NavbarToggler(id="navbar-toggler", n_clicks=0)),
                    
                    dbc.Row([
                        dbc.Col(
                            dbc.Collapse(
                                dbc.Nav([
                                    dbc.Input(type="search", placeholder="Search"),
                                    dbc.Button( "Search", color="primary", className="ms-2", n_clicks=0 ),
                                ]),
                                id="navbar-collapse", is_open=False, navbar=True
                            )
                        )
                    ],
                    align="center")
                ],
            fluid=True
            ),
    color='#0d1366', dark=True
)

button = dbc.Button( "Sleep", color="primary", className="ms-2", n_clicks=0 )

sleep = dbc.Row([
            dbc.Row([dcc.Markdown("Sleep")], style={"width": "fit-content"}),
            dbc.Row([dcc.Input(type='number', value=30, min=5, max=180, size='3', name='Sleep')], style={'width':"50%", "height":"80%"}) 
        ],
        style={'width':"185px", 'padding': '0px'}
)

def connection_status(time):
    if (time > timedelta(minutes=30)):
        return ("Lost")
    elif (time > timedelta(minutes=5)):
        return ("Unstable")
    else:
        return ("Good")

def cam_table(cams):

    sources = []
    connec = []
    detect = []
    sleep_array = []
    stopped = []

    i = 1
    for cam in cams:
        sources.append(
            html.Div([
                html.A(
                    f"Cam {i}",
                    id=f"hover-target{i}",
                    href=f"/cam{i}",
                ),
                dbc.Popover(
                    html.Img(src=dash.get_asset_url(f"cam{i}.jpg"), height="400px", alt="cam preview"),
                    target=f"hover-target{i}",
                    #body=True,
                    trigger="hover",
                ),
            ])
        )
        connec.append(connection_status(time_diff(cam)))
        detect.append(number_detections(cam))
        stopped.append(time_stopped(cam))
        sleep_array.append(dbc.Button( f"Sleep {i}", color="primary", id=f"b {i}", n_clicks=0 ))
        i += 1

    df2 = pd.DataFrame(
        {
            "Source": sources,
            "Connection": connec,
            "Detections": detect,
            "Time Stopped (s)": stopped,
            sleep: sleep_array,
        }
    )

    table = dbc.Table.from_dataframe(df2, striped=True, bordered=True, hover=True, size='sm')
    return (table)

table = cam_table(cams)

sources = []
detect = []
stopped = []
i = 1
for cam in cams:
    sources.append(f"Cam {i}" ),
    detect.append(number_detections(cam))
    stopped.append(time_stopped(cam))
    i += 1
avg = avg_detections(cams[0])
color_b = ['#0d1366'] * len(cams)
color_r = ['#0d6efd'] * len(cams)
fig = dcc.Graph(
    figure={
        'data': [
            {'x': sources, 'y': detect, 'type': 'bar', 'name': 'Nº of Jams', 
             'marker': {
               'color': color_b}, 'ybar': ''},
            {'x': sources, 'y': stopped, 'type': 'bar', 'name': 'Time Jammed (s)', 
             'marker': {
               'color': color_r}},
        ],
        'layout': {
            'title': 'Jams Detected'
        }
    },
    responsive=True,
    #style={"width": "fit-content"}
)
card = dbc.Card(
    dbc.CardBody(
        [
            fig
        ]
    )
)
app.layout = html.Div([
    navbar, 
    dbc.Container([
        dbc.Row([
            dcc.DatePickerSingle(
                calendar_orientation='vertical',
                placeholder='Select a date',
                date=date(2024, 1, 15),
                display_format= "DD/MM/YYYY"),
            table,
            card
        ], style={"text-align": "center"}),
    ]),
],style={"text-align": "center"})


'''dbc.Row([
    dbc.Col([
        html.Img(src=dash.get_asset_url("hist.png"), alt="hist")
    ], width=12),
]),'''
'''
@app.callback(
    Output(gr, component_property='figure'),
    Input(price_slider, 'value')
)


def update_graph(prices_value):
    print(prices_value)
    dff = df[df.minimum_nights >= 0]
    dff = dff[(dff.price > 50) & (dff.price < 2000)]

    fig = px.scatter_mapbox(data_frame=dff, lat='latitude', lon='longitude', color='price', height=600,
                            range_color=[0, 1000], zoom=11, color_continuous_scale=px.colors.sequential.Sunset,
                            hover_data={'latitude': False, 'longitude': False, 'room_type': True,
                                        'minimum_nights': True})
    fig.update_layout(mapbox_style='carto-positron')

    return fig


app = Flask(__name__)

# Function to pause for 30 minutes
def wait_30_minutes():
    print("Pausing for 30 minutes...")
    time.sleep(1800)
    print("Resuming after 30 minutes.")

# Route for the dashboard page
@app.route('/')
def index():
    return render_template('index.html')

# Route to trigger the pause
@app.route('/pause')
def pause():
    # Start the function in a separate thread to not block the Flask server
    threading.Thread(target=wait_30_minutes).start()
    return "Pausing for 30 minutes..."
'''
if __name__ == '__main__':
    app.run_server(debug=True)