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

class StockDashApp:
    def __init__(self, wrds_username):
        self.market_utils = MarketUtilities(wrds_username)
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def get_interday_stock_data(self, symbol, start_date, end_date):
        df = self.market_utils.interday_df_w_dates(symbol, before_dt=start_date, after_dt=end_date)
        return df
    
    def get_intraday_stock_data(self, symbol, current_dt):
        df = self.market_utils.intraday_df_w_dates(symbol=symbol, before_dt=current_dt, after_dt=current_dt, use_sqlite=False, write_sqlite=False)
        return df


    def setup_layout(self):
        self.app.layout = html.Div([
            # Top bar inputs
            # html.Div([
            #     dcc.Input(id='symbol', type='text', value='SAVA', placeholder='Enter Symbol'),
            #     dcc.DatePickerSingle(
            #         id='date-picker',
            #         date=pd.to_datetime("2022-01-01"),
            #     ),
            #     dcc.Input(id='days_bf_aft', type='number', min=1, max=30, step=1, value=15),
            #     html.Button('Run', id='run-button', n_clicks=0)
            # ], style={'display': 'flex', 'gap': '10px'}),

            # Top bar inputs
            html.Div([
                html.Div([
                    html.Label("Symbol:  ", style={'margin-right': '10px', 'font-family': 'Arial, Calibri, sans-serif'}),
                    dcc.Input(id='symbol', type='text', value='SAVA', placeholder='Enter Symbol',  style={'height': '40px', 'font-family': 'Arial, Calibri, sans-serif'})
                ], style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'center', 'margin-right': '10px'}),
                
                html.Div([
                    html.Label("Date:  ", style={'margin-right': '10px', 'font-family': 'Arial, Calibri, sans-serif'}),
                    dcc.DatePickerSingle(
                        id='date-picker',
                        date=pd.to_datetime("2022-01-01")                    )
                ], style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'center', 'margin-right': '10px'}),
                
                html.Div([
                    html.Label("Days Before/After:  ", style={'margin-right': '10px', 'font-family': 'Arial, Calibri, sans-serif'}),
                    dcc.Input(id='days_bf_aft', type='number', min=1, max=30, step=1, value=15, style={'height': '40px', 'font-family': 'Arial, Calibri, sans-serif'})
                ], style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'center', 'margin-right': '10px'}),
                
                html.Button('Run', id='run-button', n_clicks=0)
            ], style={'display': 'flex', 'gap': '10px'}),
            
            # Main chart and right-side panel
            html.Div([
                html.Div([
                    # Interday chart
                    dcc.Graph(id='interday-chart', style={'height': '600px', 'margin-bottom': '10px'}),
                    # Interday volume bar chart
                    dcc.Graph(id='interday-volume-chart', style={'height': '300px', 'margin-top': '10px'})
                ], style={'width': '80%'}),
                
                # Metrics panel
                html.Div(id='metrics-panel', style={
                    'width': '20%',
                    'padding': '10px',
                    'border': '1px solid #ccc',
                    'border-radius': '5px',
                    'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
                    'background-color': '#f9f9f9',
                    'margin': '10px',
                    'font-family': 'Arial, Calibri, sans-serif'
                })
            ], style={'display': 'flex'}),
            
            html.Div([ 
                html.Div([
                # Intraday zoom chart
                dcc.Graph(id='intraday-chart'),
                # Intraday cumulative volume chart
                dcc.Graph(id='intraday-cumulative-volume-chart'),
                # Sequential returns chart
                dcc.Graph(id='seq-percent-returns'),
                # Rolling standard deviation chart
                dcc.Graph(id='rolling-std')
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
            [Output('interday-chart', 'figure'), Output('interday-volume-chart', 'figure'), Output('metrics-panel', 'children')],
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
            price_fig = go.Figure(data=[go.Candlestick(x=df['dlycaldt'],
                                    open=df['dlyopen'],
                                    high=df['dlyhigh'],
                                    low=df['dlylow'],
                                    close=df['dlyclose'])])
            # Add title to the figure
            price_fig.update_layout(
                title=f'{symbol} Interday Candlestick Chart'
            )
            volume_fig = px.bar(df, x='dlycaldt', y='dlyvol', title=f'{symbol} Volume Over Time')

            metrics = html.Div([
                html.H3("Daily Summary Statistics", style={'text-align': 'center'}),
                html.P([
                    html.Strong("Mean Daily High: "),
                    f"{df['dlyhigh'].mean():.2f}"
                ]),
                html.P([
                    html.Strong("Mean Volume: "),
                    f"{df['dlyvol'].mean():.2f}"
                ])
            ])
            return price_fig, volume_fig, metrics

        @self.app.callback(
            [Output('intraday-chart', 'figure'), Output('intraday-cumulative-volume-chart', 'figure'), 
             Output('seq-percent-returns', 'figure'), Output('rolling-std', 'figure'),
             Output('metrics-intraday-panel', 'children')],
            [Input('interday-chart', 'clickData'), Input('symbol', 'value')]
        )
        def update_intraday_charts(clickData, symbol):
            if not clickData:
                return dash.no_update
            
            clicked_date = clickData['points'][0]['x']  # Extract clicked date
            clicked_date = pd.to_datetime(clicked_date).strftime('%Y-%m-%d')
            df = self.get_intraday_stock_data(symbol, clicked_date).sort_values("trunc_time")  # Load intraday data
            
            #price_fig = px.line(df, x='trunc_time', y='avg_price', title=f'Intraday {clicked_date}')
            #df['cumulative_volume'] = df['volume'].cumsum()
            #volume_fig = px.line(df, x='trunc_time', y='cumulative_volume', title=f'Cumulative Volume {clicked_date}')
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

            # Create the price figure
            price_fig = px.line(df, x='trunc_time', y='avg_price', title=f'Intraday {clicked_date}')

            # Create the cumulative volume figure
            volume_fig = px.line(df, x='trunc_time', y='cumulative_volume', title=f'Cumulative Volume {clicked_date}')

            # line plot for returns
            seq_returns_fig = px.line(df, x='trunc_time', y='returns', title=f'Sequential Returns {clicked_date}')

            # line plot for rolling standard deviation of returns
            rolling_std_fig = px.line(df, x='trunc_time', y='rolling_std_returns', title=f'Rolling Standard Deviation of Returns {clicked_date}')
            
            metrics = html.Div([
                html.P(f"Mean Price: {df['avg_price'].mean():.2f}"),
                html.P(f"Total Volume: {df['volume'].sum()}"),
                html.P(f"Max Price: {df['avg_price'].max():.2f}"),
                html.P(f"Min Price: {df['avg_price'].min():.2f}")
            ])
            
            return price_fig, volume_fig, seq_returns_fig, rolling_std_fig, metrics

    def run(self):
        self.app.run_server(debug=True)

# Example usage
if __name__ == '__main__':
    app = StockDashApp(wrds_username="audreymcmillion")
    app.run()