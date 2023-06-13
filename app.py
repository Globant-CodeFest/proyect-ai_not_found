import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import requests

# Preparar datos
url = 'https://1sxs2ownii.execute-api.us-east-1.amazonaws.com/qa/predict_service'

try:
    response = requests.post(url)
    response.raise_for_status()
    response_data = response.json()
    data = response_data['body']
except Exception as e:
    print(f"No se pudieron obtener los datos de la API: {e}")
    data = [
        {"id": 0, "date": "2020-08-13", "value": 25479.476, "currency": "BTC"},
        {"id": 1, "date": "2020-08-14", "value": 27462.788, "currency": "BTC"},
        # Añade aquí más datos para ETH y BNB
    ]

df = pd.DataFrame(data)

# Crear aplicación Dash
app = dash.Dash(__name__)
server = app.server  # Exponer el servidor Flask para poder agregar rutas

# Crear el layout de la aplicación
app.layout = html.Div(children=[
    html.H1(children='Gráfico Dinámico en Dash'),

    dcc.Dropdown(
        id='currency-dropdown',
        options=[
            {'label': 'Bitcoin (BTC)', 'value': 'BTC'},
            {'label': 'Ethereum (ETH)', 'value': 'ETH'},
            {'label': 'Binance Coin (BNB)', 'value': 'BNB'},
        ],
        value='BTC'
    ),

    dcc.Input(
        id='days-input',
        type='number',
        min=1,
        value=7
    ),

    dcc.Graph(id='example-graph')
])

@app.callback(
    Output('example-graph', 'figure'),
    [Input('currency-dropdown', 'value'),
     Input('days-input', 'value')]
)
def update_graph(selected_currency, number_of_days):
    filtered_df = df[(df['currency'] == selected_currency) & (df['id'] < number_of_days)]
    return {
        'data': [
            go.Scatter(
                x=filtered_df['date'],
                y=filtered_df['value'],
                mode='lines+markers'
            )
        ],
        'layout': go.Layout(title=f'Valores de {selected_currency} a lo largo de {number_of_days} días')
    }


if __name__ == '__main__':
    app.run_server(debug=True)
