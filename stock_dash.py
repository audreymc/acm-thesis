# DISCLAIMER: Developed with the help of ChatGPT o3 and MS Copilot
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import sqlite3
from market_utils import MarketUtilities
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
import plotly.io as pio

class StockDashApp:
    def __init__(self, wrds_username):
        self.market_utils = MarketUtilities(wrds_username)
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.setup_layout()
        self.setup_callbacks()
        pio.templates.default = "plotly_white"

    def get_interday_stock_data(self, symbol, start_date, end_date):
        df = self.market_utils.interday_df_w_dates(symbol, before_dt=start_date, after_dt=end_date)
        return df
    
    def get_intraday_stock_data(self, symbol, current_dt):
        df = self.market_utils.intraday_df_w_dates(symbol=symbol, before_dt=current_dt, after_dt=current_dt, use_sqlite=False, write_sqlite=False)
        return df


    def setup_layout(self):
        self.app.layout = html.Div([
            # Top bar inputs
            html.Div([
                html.Div([
                    html.Label("Symbol:  ", style={'margin-right': '10px'}),
                    dcc.Input(id='symbol', type='text', value='SAVA', placeholder='Enter Symbol',  style={'height': '40px'})
                ], style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'center', 'margin-right': '10px', 'margin-left': '10px'}),
                
                html.Div([
                    html.Label("Date:  ", style={'margin-right': '10px'}),
                    dcc.DatePickerSingle(
                        id='date-picker',
                        date=pd.to_datetime("2022-01-01")
                    )
                ], style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'center', 'margin-right': '10px'}),
                
                html.Div([
                    html.Label("Days Before/After:  ", style={'margin-right': '10px'}),
                    dcc.Input(id='days_bf_aft', type='number', min=1, max=30, step=1, value=15, style={'height': '40px'})
                ], style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'center', 'margin-right': '10px'}),
                
                html.Button('Run', id='run-button', n_clicks=0)
            ], style={
                'display': 'flex',
                'gap': '10px',
                'align-items': 'center',
                'margin-top': '30px',
                'margin-bottom': '20px',
                # 'font-family': 'Arial, Calibri, sans-serif'
            }),

            html.Div([
                html.H2("Interday Price and Volume", style={
                    'text-align': 'center',
                    # 'font-family': 'Arial, Calibri, sans-serif',
                    'font-weight': 'bold',
                    'margin-bottom': '10px'
                }),
                html.P(
                    "These charts show the interday price and volume for a given stock symbol. You can click on a date in the interday price chart to see intraday details below.",
                    style={
                        'text-align': 'center',
                        'font-family': 'Arial, Calibri, sans-serif',
                        'font-size': '18px',
                        'margin-bottom': '20px'
                    }
                ),
            ], style={
                'display': 'flex',
                'flex-direction': 'column',
                'align-items': 'center',
                'justify-content': 'center',
                'width': '100%',
                'margin-bottom': '20px'
            }),

            # Main chart and right-side panel
            html.Div([
                html.Div([
                    # Interday chart
                    dcc.Graph(id='interday-chart', style={'height': '600px', 'margin-bottom': '10px'})
                ], style={'width': '80%'}),
                
                # Metrics panel
                html.Div(id='metrics-panel', style={
                    'width': '20%',
                    'padding': '10px',
                    'border': '1px solid #ccc',
                    'border-radius': '5px',
                    'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
                    'background-color': '#f9f9f9',
                    'margin': '10px'
                })
            ], style={'display': 'flex'}),

            html.Div([
                html.H2("Intraday Price and Volume", style={
                    'text-align': 'center',
                    # 'font-family': 'Arial, Calibri, sans-serif',
                    'font-weight': 'bold',
                    'margin-bottom': '10px'
                }),
            ], style={
                'display': 'flex',
                'flex-direction': 'column',
                'align-items': 'center',
                'justify-content': 'center',
                'width': '100%',
                'margin-bottom': '20px',
                'margin-top': '20px'
            }),
            
            html.Div([ 
                html.Div([
                # Intraday zoom chart
                dcc.Graph(id='intraday-chart')
                ], style={'width': '80%'}),

                # Metrics panel
                html.Div(id='metrics-intraday-panel', style={
                    'width': '20%',
                    'padding': '10px',
                    'border': '1px solid #ccc',
                    'border-radius': '5px',
                    'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
                    'background-color': '#f9f9f9',
                    'margin': '10px'
                })
            ], style={'display': 'flex'})
        ])

    def setup_callbacks(self):
        @self.app.callback(
            [Output('interday-chart', 'figure'), Output('metrics-panel', 'children')],
            [Input('run-button', 'n_clicks')],
            [State('symbol', 'value'),
             State('date-picker', 'date'),
             State('days_bf_aft', 'value')]
        )
        def update_interday_charts(n_clicks, symbol, current_dt, diff_num):
            if n_clicks == 0:
                return dash.no_update
    
            current_dt = pd.to_datetime(current_dt).strftime('%Y-%m-%d')
            df = self.get_interday_stock_data(symbol, 
                                              self.market_utils.get_before_date(current_dt, diff_num), 
                                              self.market_utils.get_after_date(current_dt, diff_num))
            # Create a single figure with two subplots sharing the x-axis
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(
                    f'{symbol} Interday Candlestick Chart',
                    f'{symbol} Volume Over Time'
                ),
                row_heights=[0.7, 0.3]
            )

            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df['dlycaldt'],
                    open=df['dlyopen'],
                    high=df['dlyhigh'],
                    low=df['dlylow'],
                    close=df['dlyclose'],
                    name='Price'
                ),
                row=1, col=1
            )
            fig.update_layout(xaxis_rangeslider_visible=False)

            # Volume bar chart
            fig.add_trace(
                go.Bar(
                    x=df['dlycaldt'],
                    y=df['dlyvol'],
                    name='Volume',
                    marker_color='rgba(50, 150, 255, 0.5)'
                ),
                row=2, col=1
            )

            fig.update_layout(
                height=600,
                showlegend=False,
                title_text=f'{symbol} Interday Price & Volume',
                margin=dict(t=60, b=40, l=40, r=20)
            )

            price_fig = fig

            metrics = html.Div([
                html.H4("Interday Statistics", style={'text-align': 'center', 'margin-bottom': '10px'}),
                html.P([
                    html.Strong("Mean Daily High: "),
                    f"${df['dlyhigh'].mean():.2f}"
                ]),
                html.P([
                    html.Strong("Mean Volume: "),
                    f"{df['dlyvol'].mean():,.2f}"
                ])
            ], style={'margin-left': '10px', 'font-size': '16px'})
            return price_fig, metrics

        @self.app.callback(
            [Output('intraday-chart', 'figure'),
             Output('metrics-intraday-panel', 'children')],
            [Input('interday-chart', 'clickData'), Input('symbol', 'value')]
        )
        def update_intraday_charts(clickData, symbol):
            if not clickData:
                return dash.no_update
            
            clicked_date = clickData['points'][0]['x']  # Extract clicked date
            clicked_date = pd.to_datetime(clicked_date).strftime('%Y-%m-%d')
            df = self.get_intraday_stock_data(symbol, clicked_date).sort_values("trunc_time")  # Load intraday data
            
            # get time
            df['trunc_time'] = pd.to_datetime(clicked_date + ' ' + df['trunc_time'].astype(str))

            # Generate a full range of timestamps at 1-second intervals
            full_time_range = pd.date_range(
                start=df['trunc_time'].min(),
                end=df['trunc_time'].max(),
                freq='1S'
            )

            # Calculate cumulative volume
            df['cumulative_volume'] = df['volume'].cumsum()
            
            # Calculate sequential returns
            df['returns'] = df['avg_price'].pct_change()

            # Calculate the rolling standard deviation of the returns
            df['rolling_std_returns'] = df['returns'].rolling(window=10).std()

            # Reindex the DataFrame to include all timestamps and forward-fill missing values
            df = df.set_index('trunc_time').reindex(full_time_range, method='ffill').reset_index()
            df.rename(columns={'index': 'trunc_time'}, inplace=True)
            # Create a single figure with four subplots sharing the same x-axis

            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(
                    f'Intraday Price {clicked_date}',
                    f'Cumulative Volume {clicked_date}',
                    f'Sequential Returns {clicked_date}',
                    f'Rolling Std of Returns {clicked_date}'
                ),
                row_heights=[0.3, 0.2, 0.25, 0.25]
            )

            # Price line
            fig.add_trace(
                go.Scatter(x=df['trunc_time'], y=df['avg_price'], name='Price', mode='lines'),
                row=1, col=1
            )
            # Cumulative volume line
            fig.add_trace(
                go.Scatter(
                    x=df['trunc_time'],
                    y=df['cumulative_volume'],
                    name='Cumulative Volume',
                    mode='lines',
                    line=dict(color='orange'),
                    fill='tozeroy'  # This fills the area under the line to the x-axis
                ),
                row=2, col=1
            )

            # Sequential returns line
            fig.add_trace(
                go.Scatter(x=df['trunc_time'], y=df['returns'], name='Returns', mode='lines', line=dict(color='green')),
                row=3, col=1
            )
            # Rolling std line
            fig.add_trace(
                go.Scatter(x=df['trunc_time'], y=df['rolling_std_returns'], name='Rolling Std', mode='lines', line=dict(color='red')),
                row=4, col=1
            )

            fig.update_layout(
                height=900,
                showlegend=False,
                title_text=f'Intraday Analysis for {symbol} on {clicked_date}',
                margin=dict(t=60, b=40, l=40, r=20)
            )

            price_fig = fig
            
            metrics = html.Div([
                html.H4("Intraday Summary Statistics", style={'text-align': 'center', 'margin-bottom': '10px'}),
                html.Div([
                    html.P([
                        html.Strong("Mean Price: "),
                        f"${df['avg_price'].mean():.2f}"
                    ]),
                    html.P([
                        html.Strong("Total Volume: "),
                        f"{df['volume'].sum():,.0f}"
                    ]),
                    html.P([
                        html.Strong("Max Price: "),
                        f"${df['avg_price'].max():.2f}"
                    ]),
                    html.P([
                        html.Strong("Min Price: "),
                        f"${df['avg_price'].min():.2f}"
                    ])
                ], style={'margin-left': '10px', 'font-size': '16px'})
            ], style={
                'padding': '10px'
            })
            
            return price_fig, metrics

    def run(self):
        self.app.run_server(debug=True)

# Example usage
if __name__ == '__main__':
    app = StockDashApp(wrds_username="audreymcmillion")
    # app.run_server(debug=True)
    app.run()