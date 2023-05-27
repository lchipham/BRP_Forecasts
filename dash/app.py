from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import date as dt

#----------------------------------------------------------------------------------------------------------------------#
# Application
app = Dash(__name__)
# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

# App Introduction (Markdown)
markdown_text = '''
Excess bond returns is a proxy for realized risk premium of a long-term bond. Forecasting excess bond returns is crucial for optimizing strategic asset allocation decisions, 
managing risks associated with bond holdings, developing fixed-income trading strategies, mitigating interest rate risk, 
and conducting relative value analysis within the bond market. Target variable (excess bond return) is the monthly return of the 10-year Treasury bond 
in excess of the nominally riskless return of a one-month Treasury bill. Four relevant variables are included as predictors:
'''
# Variables Info
variable_info = '''
- Term Spread (Main proxy for BRP): Difference between the estimated five-year spot rate and the three-month spot rate. 
- Real Yield: Difference between the estimated five-year spot rate and the most recently published yearly consumer price inflation rate. 
- Inverse Wealth: Ratio of the exponentially weighted past stock market level to the current stock market level.
- Momentum: Dummy variable (1: bond yield > 5pbs below 6-month average, -1: the bond yield is > 5bps above 6-month average, 0: otherwise). 
'''
#----------------------------------------------------------------------------------------------------------------------#
# Load full dataset
exc_ret_mdl = pd.read_csv('exc_ret_modeling.csv')
exc_ret_mdl['momentum_ft'] = exc_ret_mdl['momentum'].replace({1: 'long', -1: 'short'})

# Algo Predictions: Train & Test
train_preds = pd.read_csv('train_preds.csv')
train_preds['momentum'] = train_preds.momentum.astype('category')
train_preds['momentum'] = train_preds['momentum'].replace({1: 'long', -1: 'short'})

test_preds = pd.read_csv('test_preds.csv')
test_preds['momentum'] = test_preds.momentum.astype('category')
test_preds['momentum'] = test_preds['momentum'].replace({1: 'long', -1: 'short'})

# Excess bond returns in sub-samples
# Table formation
total_obs = exc_ret_mdl['term_spread'].count()
subsample1 = pd.DataFrame({
    'Months Begin With': ['Average Excess Returns', 'Share of Total Months (%)'],
    'Term Spread > 0': [round(exc_ret_mdl[exc_ret_mdl['term_spread'] > 0]['excess'].mean(), 2), round(exc_ret_mdl[exc_ret_mdl['term_spread'] > 0]['excess'].count() / total_obs * 100, 2)],
    'Term Spread < 0': [round(exc_ret_mdl[exc_ret_mdl['term_spread'] < 0]['excess'].mean(), 2), int(exc_ret_mdl[exc_ret_mdl['term_spread'] < 0]['excess'].count() / total_obs * 100)],
    'Inverse Wealth > 0.09': [round(exc_ret_mdl[exc_ret_mdl['inv_wealth'] > 0.09]['excess'].mean(), 2), int(exc_ret_mdl[exc_ret_mdl['inv_wealth'] > 0.09]['excess'].count() / total_obs * 100)],
    'Inverse Wealth < 0.09': [round(exc_ret_mdl[exc_ret_mdl['inv_wealth'] < 0.09]['excess'].mean(), 2), int(exc_ret_mdl[exc_ret_mdl['inv_wealth'] < 0.09]['excess'].count() / total_obs * 100)]
})

subsample2 = pd.DataFrame({
    'Months Begin With': ['Average Excess Returns', 'Share of Total Months (%)'],
    'Term Spread > 0 & Inverse Wealth > 0.09': [round(exc_ret_mdl[(exc_ret_mdl['term_spread'] > 0) & (exc_ret_mdl['inv_wealth'] > 0.09)]['excess'].mean(), 2), round(exc_ret_mdl[(exc_ret_mdl['term_spread'] > 0) & (exc_ret_mdl['inv_wealth'] > 0.09)]['excess'].count() / total_obs * 100, 2)],
    'Term Spread > 0 & Inverse Wealth < 0.09': [round(exc_ret_mdl[(exc_ret_mdl['term_spread'] > 0) & (exc_ret_mdl['inv_wealth'] < 0.09)]['excess'].mean(), 2), round(exc_ret_mdl[(exc_ret_mdl['term_spread'] > 0) & (exc_ret_mdl['inv_wealth'] < 0.09)]['excess'].count() / total_obs * 100, 2)],
    'Term Spread < 0 & Inverse Wealth > 0.09': [round(exc_ret_mdl[(exc_ret_mdl['term_spread'] < 0) & (exc_ret_mdl['inv_wealth'] > 0.09)]['excess'].mean(), 2), round(exc_ret_mdl[(exc_ret_mdl['term_spread'] < 0) & (exc_ret_mdl['inv_wealth'] > 0.09)]['excess'].count() / total_obs * 100, 2)],
    'Term Spread < 0 & Inverse Wealth < 0.09': [round(exc_ret_mdl[(exc_ret_mdl['term_spread'] < 0) & (exc_ret_mdl['inv_wealth'] < 0.09)]['excess'].mean(), 2), round(exc_ret_mdl[(exc_ret_mdl['term_spread'] < 0) & (exc_ret_mdl['inv_wealth'] < 0.09)]['excess'].count() / total_obs * 100, 2)]
})

# Expanding Rolling Regression - Expected Excess Bond Returns
exp_rolling_reg = pd.read_csv('exp_rolling_reg.csv')
# Time Series Plot: Predicted vs Actual Excess Bond Returns
excess = go.Scatter(x=exp_rolling_reg['DATE'], y=exp_rolling_reg['excess'],
                   name='Realized', mode='lines',
                   hovertemplate='%{x|%Y-%m-%d}: %{y:.2f}')
exp_excess = go.Scatter(x=exp_rolling_reg['DATE'], y=exp_rolling_reg['expected_excess'],
                   name='Predicted', mode='lines',
                   hovertemplate='%{x|%Y-%m-%d}: %{y:.2f}')
layout0 = go.Layout(title=dict(text='Predicted vs Realized Excess Bond Returns', font=dict(family='Serif', size=22)),
                   xaxis=dict(title=None), yaxis=dict(title=None))
rolling_excess = go.Figure(data=[excess, exp_excess], layout=layout0)
rolling_excess.update_layout(plot_bgcolor='white', paper_bgcolor='white')
rolling_excess.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
rolling_excess.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
rolling_excess.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

# Time Series of Predictors (see https://plotly.com/python/px-arguments/ for more options)
pred1 = go.Scatter(x=exc_ret_mdl['DATE'], y=exc_ret_mdl['term_spread'],
                   name='Term Spread', mode='lines',
                   hovertemplate='%{x|%Y-%m-%d}: %{y:.2f}')
pred2 = go.Scatter(x=exc_ret_mdl['DATE'], y=exc_ret_mdl['real_yield'],
                   name='Real Yield', mode='lines',
                   hovertemplate='%{x|%Y-%m-%d}: %{y:.2f}')
pred3 = go.Scatter(x=exc_ret_mdl['DATE'], y=exc_ret_mdl['inv_wealth'],
                   name='Inverse Wealth', mode='lines',
                   hovertemplate='%{x|%Y-%m-%d}: %{y:.2f}')
pred4 = go.Scatter(x=exc_ret_mdl['DATE'], y=exc_ret_mdl['momentum'],
                   name='Momentum', mode='lines',
                   hovertemplate='%{x|%Y-%m-%d}: %{y:.2f}')
layout = go.Layout(title=dict(text='Historical Levels of Predictors', font=dict(family='Serif', size=22)),
                   xaxis=dict(title=None), yaxis=dict(title=None))
ts_preds = go.Figure(data=[pred1, pred2, pred3, pred4], layout=layout)
ts_preds.update_layout(plot_bgcolor='white', paper_bgcolor='white')
ts_preds.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
ts_preds.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
ts_preds.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)

# Backtesting Results: Dynamic & Static Investment Strategies
backtest = pd.read_csv('backtest_results.csv')
backtest = backtest.iloc[:,1:]

# Backtest Technicals
backtest_info = '''
1. Dynamic
- Scale: Long/short long-term bond in proportion to the size of predicted BRP (bond risk premium).
- 1/10: Buy 1 unit of bond when BRP > 0, 0 when BRP < 0.
2. Static:
- Always-bond: always holding 10-year Treasury bond
- Bond-cash: 50% in cash (1-month T-Bill), 50% in bond (10-year T-Note)
'''
#----------------------------------------------------------------------------------------------------------------------#
# APPLICATION LAYOUT
app.layout = html.Div(children=[
    # NAVIGATION BAR

    # TITLE
    html.H2(children='Forecasting Realized Bond Risk Premium',
            style={'textAlign': 'center'}),
    html.Div([
        dcc.Markdown(children=markdown_text, style={'textAlign': 'left'})
    ]),

    # Variable Info & Time Series
    html.Div([
        dcc.Markdown(children=variable_info, style={'textAlign': 'left'})
    ]),
    html.Div([
        dcc.Graph(
            id='ts-predictors',
            figure=ts_preds)
    ]),

    # SUMMARY TABLE
    html.H3(children='Excess Bond Returns in Subsamples',
            style={'textAlign': 'left'}),
    html.Div([
        dash_table.DataTable(
            data=subsample1.to_dict('records'),
            columns=[{'name': col, 'id': col} for col in subsample1.columns]),
        dash_table.DataTable(
            data=subsample2.to_dict('records'),
            columns=[{'name': col, 'id': col} for col in subsample2.columns])
    ], style={'display': 'inline-block'}),


    # Section I Title
    #html.Br(),
    html.H3(children='Train - Test Prediction',
            style={'textAlign': 'left'}),

    # USER INPUT
    html.Div([
        html.Div([
            dcc.Dropdown(['Multiple Linear Regression', 'Random Forest', 'Gradient Boosting', 'K-Nearest Neighbor'],
                         value='Multiple Linear Regression',
                         #style={'width': '30%'},
                         id='algo-option'),
            dcc.Dropdown(['Multiple Linear Regression', 'Random Forest', 'Gradient Boosting', 'K-Nearest Neighbor'],
                         multi=True,
                         #style={'width': '30%'},
                         id='combine-algo'),
            html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
            html.Div(id='output-state')
        ], style={'width': '50%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                options=[{'label': 'Long', 'value': 'long'},
                         {'label': 'Short', 'value': 'short'},
                         {'label': 'Long/Short', 'value': '0'}],
                value='0',
                #style={'width': '60%'},
                id='ls_position')
            # dcc.DatePickerRange(min_date_allowed=dt(2002, 2, 1),
            #                     max_date_allowed=dt(2023, 2, 1),
            #                     #initial_visible_month=dt.today(),
            #                     #style={'height': '10px', 'font-size': '8px'}),
            #                     id='date-range')
        ], style={'width': '50%', 'display': 'inline-block'})
    ]),

    # OUTPUT
    html.Div([
        dcc.Graph(id='perf_scatter')
    ], style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),

    html.Div([
        dcc.Graph(id='corr_bar')
        #dcc.Graph(id='corr_heatmap')
    ], style={'display': 'inline-block', 'width': '50%'}),

    # Section II Title
    #html.Br(),
    html.H3(children='Monthly Rolling Prediction (MLR)',
            style={'textAlign': 'left'}),
    # Rolling Reg
    html.Div([
        dcc.Graph(
            id='rolling-reg',
            figure=rolling_excess)
    ]),
    # Section III Title
    #html.Br(),
    html.H3(children='Backtesting on Dynamic & Static Strategies',
            style={'textAlign': 'left'}),
    html.Div([
            dcc.Markdown(children=backtest_info, style={'textAlign': 'left'})
        ]),
    # BACKTEST TABLE
    html.Div([
        dash_table.DataTable(
            data=backtest.to_dict('records'),
            columns=[{'name': col, 'id': col} for col in backtest.columns])
    ], style={'width': '92%', 'display': 'inline-block'}),
])

@app.callback(
    Output('perf_scatter', 'figure'),
    Input('algo-option', 'value'),
    Input('ls_position', 'value'))
def update_graph(algo_name, strategy):
    if strategy != '0':
        train_sample = train_preds[train_preds['momentum'] == strategy]
        test_sample = test_preds[test_preds['momentum'] == strategy]
    else:
        train_sample = train_preds
        test_sample = test_preds

    # Train & Test MSE
    train_mse = round(np.mean(np.abs(np.array(train_sample[algo_name]) - np.array(train_sample['train_actuals']))), 4)
    #train_rmse = np.sqrt(np.mean((np.array(train_preds) - np.array(train_actuals)) ** 2))
    test_mse = round(np.mean(np.abs(np.array(test_sample[algo_name]) - np.array(test_sample['test_actuals']))), 4)
    #test_rmse = np.sqrt(np.mean((np.array(test_preds) - np.array(test_actuals))  ** 2))

    # Plot performance
    trace1 = go.Scatter(
        x=train_sample[algo_name],
        y=train_sample['train_actuals'],
        mode='markers',
        name='Train',
        text=train_sample['date'],
        customdata=round(train_sample['train_actuals'] - train_sample[algo_name],4),
        hovertemplate='<b>Prediction:</b> %{x}<br>' +
                      '<b>Actual:</b> %{y}<br>' +
                      '<b>Diff:</b> %{customdata}<br>' +
                      '<b>Date:</b> %{text}<br>'
    )
    trace2 = go.Scatter(
        x=test_sample[algo_name],
        y=test_sample['test_actuals'],
        mode='markers',
        name='Test',
        text=test_sample['date'],
        customdata=round(test_sample['test_actuals'] - test_sample[algo_name], 4),
        hovertemplate='<b>Prediction:</b> %{x}<br>' +
                      '<b>Actual:</b> %{y}<br>' +
                      '<b>Diff:</b> %{customdata}<br>' +
                      '<b>Date:</b> %{text}<br>'
    )
    # create figure with two traces
    fig = go.Figure(data=[trace1, trace2])
    fig.update_layout(
        title={'text': algo_name + ' Performance', 'font': {'family': 'Serif', 'size': 22}},
        xaxis={'title': {'text': 'Predictions', 'font': {'family': 'Serif', 'size': 18}}},
        yaxis={'title': {'text': 'Actuals', 'font': {'family': 'Serif', 'size': 18}}},
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.add_annotation(
        x=2.6, y=1, align='center',
        text='Train MSE: ' + str(train_mse) + '   Test MSE: ' + str(test_mse),
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, ax=30, ay=30,
        font=dict(size=14,color='black',family='Serif'),
        bordercolor='black', borderwidth=1, borderpad=4
    )
    return fig

@app.callback(
    Output('corr_bar', 'figure'),
    Input('ls_position', 'value')
)
def update_bar(strategy):
    # LS filter
    if strategy != '0':
        exc_ret_sample = exc_ret_mdl[exc_ret_mdl['momentum_ft'] == strategy]
    else:
        exc_ret_sample = exc_ret_mdl

    # Calculate correlation matrix
    corr_matrix = exc_ret_sample.drop('rolling_6m', axis=1).corr(numeric_only=True)

    # Create bar graph of predictors' correlation to target
    full_matrix = corr_matrix.reset_index(drop=False)  # Reset index
    full_matrix['index'] = full_matrix['index'].replace({'term_spread': 'Term Spread', 'real_yield': 'Real Yield', 'inv_wealth': 'Inverse Wealth', 'momentum': 'Momentum'})
    target_corr = px.bar(full_matrix.iloc[1:,], x='index', y='excess', color='index')
    target_corr.update_layout(
        title={'text': 'Predictors Correlation to Excess Returns', 'font': {'family': 'Serif', 'size': 20}},
        xaxis={'title': None},
        yaxis={'title': {'text': 'Correlation', 'font': {'family': 'Serif', 'size': 18}}},
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    target_corr.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    target_corr.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    return target_corr

# @app.callback(
#     Output('corr_heatmap', 'figure'),
#     Input('ls_position', 'value')
# )
# def update_hmap(strategy):
#     # LS filter
#     if strategy != '0':
#         exc_ret_sample = exc_ret_mdl[exc_ret_mdl['momentum_ft'] == strategy]
#     else:
#         exc_ret_sample = exc_ret_mdl
#     # Calculate correlation matrix
#     corr_matrix = exc_ret_sample.drop('rolling_6m', axis=1).corr(numeric_only=True)
#     # Create correlation heatmap
#     hmap = px.imshow(corr_matrix, labels=dict(x="Feature 1", y="Feature 2", color="Correlation"),
#                     x=corr_matrix.columns, y=corr_matrix.columns)
#     return hmap

if __name__ == '__main__':
    app.run_server(debug=True)
