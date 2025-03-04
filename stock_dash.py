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
            html.Div([
                dcc.Input(id='symbol', type='text', value='SAVA', placeholder='Enter Symbol'),
                dcc.DatePickerSingle(
                    id='date-picker',
                    date=pd.to_datetime("2022-01-01"),
                ),
                dcc.Input(id='days_bf_aft', type='number', min=1, max=30, step=1, value=15),
                html.Button('Run', id='run-button', n_clicks=0)
            ], style={'display': 'flex', 'gap': '10px'}),
            
            # Main chart and right-side panel
            html.Div([
                html.Div([
                    # Interday chart
                    dcc.Graph(id='interday-chart', style={'height': '600px', 'margin-bottom': '10px'}),
                    # Interday volume bar chart
                    dcc.Graph(id='interday-volume-chart', style={'height': '300px', 'margin-top': '10px'})
                ], style={'width': '70%'}),
                
                # Metrics panel
                html.Div(id='metrics-panel', style={
                    'width': '30%',
                    'padding': '10px',
                    'border': '1px solid #ccc',
                    'border-radius': '5px',
                    'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
                    'background-color': '#f9f9f9',
                    'margin': '10px'
                })
            ], style={'display': 'flex'}),
            
            html.Div([ 
                html.Div([
                # Intraday zoom chart
                dcc.Graph(id='intraday-chart'),
                # Intraday cumulative volume chart
                dcc.Graph(id='intraday-cumulative-volume-chart')
                ], style={'width': '70%'}),

                # Metrics panel
                html.Div(id='metrics-intraday-panel', style={
                    'width': '30%',
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
            volume_fig = px.bar(df, x='dlycaldt', y='dlyvol', title=f'{symbol} Volume Over Time')

            metrics = html.Div([
                html.P(f"Mean Daily High: {df['dlyhigh'].mean():.2f}"),
                html.P(f"Mean Volume: {df['dlyvol'].mean()}")
            ])
            return price_fig, volume_fig, metrics

        @self.app.callback(
            [Output('intraday-chart', 'figure'), Output('intraday-cumulative-volume-chart', 'figure'), Output('metrics-intraday-panel', 'children')],
            [Input('interday-chart', 'clickData'), Input('symbol', 'value')]
        )
        def update_intraday_charts(clickData, symbol):
            if not clickData:
                return dash.no_update
            
            clicked_date = clickData['points'][0]['x']  # Extract clicked date
            clicked_date = pd.to_datetime(clicked_date).strftime('%Y-%m-%d')
            df = self.get_intraday_stock_data(symbol, clicked_date).sort_values("trunc_time")  # Load intraday data
            
            price_fig = px.line(df, x='trunc_time', y='avg_price', title=f'Intraday {clicked_date}')
            df['cumulative_volume'] = df['volume'].cumsum()
            volume_fig = px.line(df, x='trunc_time', y='cumulative_volume', title=f'Cumulative Volume {clicked_date}')
            
            metrics = html.Div([
                html.P(f"Mean Price: {df['avg_price'].mean():.2f}"),
                html.P(f"Total Volume: {df['volume'].sum()}"),
                html.P(f"Max Price: {df['avg_price'].max():.2f}"),
                html.P(f"Min Price: {df['avg_price'].min():.2f}")
            ])
            
            return price_fig, volume_fig, metrics

    def run(self):
        self.app.run_server(debug=True)

# Example usage
if __name__ == '__main__':
    app = StockDashApp(wrds_username="audreymcmillion")
    app.run()