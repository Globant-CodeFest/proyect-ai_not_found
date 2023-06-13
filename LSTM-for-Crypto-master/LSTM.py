# Importar librerias a usar
import numpy as np
import time as tm
import datetime as dt
import tensorflow as tf
import pandas as pd

import requests
import json

# Preparación de los datos
from sklearn.preprocessing import MinMaxScaler

# modelo de ML
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# librería gráfica
import matplotlib.pyplot as plt



import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error


def calcular_dias_en_n_anios(n):
    cantidad_dias = n * 365  # Cantidad de días por año sin considerar años bisiestos
    cantidad_bisiestos = n // 4 - n // 100 + n // 400  # Cantidad de años bisiestos en n años
    cantidad_dias += cantidad_bisiestos  # Se suman los días adicionales por los años bisiestos
    return cantidad_dias

def plot_history(history, width=12, height=6):
  """
  DESCRIPTION:
    History performance of the keras model
  
  INPUT:
    @param history: history of performance of fitted model
    @type history: tensorflow.python.keras.callbacks.History

  OUTPUT:
    A graphic
  """

  ## Metrics keys stored in tensorflow object
  keys = list(history.history.keys())

  ## Number of epoch used for fit the model
  epoch = range(1, len(history.epoch) +1)

  ## Check if validation set was used.
  withValidation = False
  for key in keys:
    if 'val' in key:
      withValidation = True

  ## Number of metrics 
  nMetrics = len(keys)
  if withValidation:
    nMetrics = nMetrics//2

  ## Plot-space instance
  plt.figure(figsize=(width, height))

  for i in range(nMetrics):
    plt.subplot(nMetrics, 1, i+1)

    ## Plot (train) metric value
    labelMetric = keys[i]
    metric = history.history[keys[i]]
    plt.plot(epoch, metric, 'o-', label=labelMetric)

    if withValidation:
      ## Plot (validation) metric value
      labelMetricVal = keys[i+nMetrics]
      metricVal = history.history[keys[i+nMetrics]]
      plt.plot(epoch, metricVal, 'o-', label=labelMetricVal)

    plt.xlim(epoch[0], epoch[-1])
    plt.legend()
    plt.grid()

  plt.xlabel('Epoch')
  plt.show()


# Variables de entrada
n = 5  # Cantidad de años
dias_totales = calcular_dias_en_n_anios(n)

# Tamaño de la ventana, 7 (1 semana)
N_STEPS = 7

# Stock ticker, Bitcoin
STOCK = 'Bitcoin'

date_now = tm.strftime('%Y-%m-%d')

# 1829 días son años considerando años biciestos
date_n_years_back = (dt.date.today() - dt.timedelta(days = dias_totales)).strftime('%Y-%m-%d')

INTERVAL = '1d'

# size of test
TEST_SIZE = 0.3

# batch size
BATCH_SIZE = 32

# Lectura de los datos
# Especifica la ruta y nombre del archivo CSV
csv_file = f'csv_files/coin_{STOCK}.csv'

# Define las fechas de inicio y fin
fecha_inicio = date_n_years_back

# Lee el archivo CSV en un DataFrame
df = pd.read_csv(csv_file)

# Convierte la columna de fecha a tipo datetime
df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

# Filtra el DataFrame según la fechas de inicio
df_filtrado = df[(df['Date'] >= fecha_inicio)]

# Borrar columnas que no se usan
df_filtrado = df_filtrado.drop(['SNo','Name','Symbol','High','Low','Open','Volume','Marketcap'], axis=1)
df_filtrado['Close'] = df_filtrado['Close'].astype('float32')

# Reemplazar el index por la columna de fecha
df_filtrado.index = df_filtrado['Date']
df_filtrado = df_filtrado.drop(['Date'], axis=1)

# Obtener los datos de entrenamiento y de prueba
# Split into train and test sets
test_size = int(len(df_filtrado) * TEST_SIZE)
train_size = len(df_filtrado) - test_size

df_train = df_filtrado.iloc[:train_size, :]
df_test = df_filtrado.iloc[train_size:,:]

# Convertir el DataFrame a un array de numpy
dataset_train, dataset_test = df_train.values, df_test.values


#Escalado de los datos
scaler = MinMaxScaler()
ds_scale_train = scaler.fit_transform(dataset_train)
ds_scale_test = scaler.transform(dataset_test)

# Crear la función para generar las características y etiquetas
def create_features_and_target(dataset, look_back = N_STEPS):
  dataX, dataY = [], []
  for i in range(len(dataset)-look_back -1):
    a = dataset[i:(i+look_back), 0]
    dataX.append(a)
    dataY.append(dataset[i + look_back, 0])
  return np.array(dataX), np.array(dataY)

trainX, trainY = create_features_and_target(ds_scale_train)
testX, testY = create_features_and_target(ds_scale_test)

# Crear EarlyStopping

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=5,
                                                    mode='min')

# Crear el modelo
# Mejores parámetros
best_params = {'num_units_lstm1': 403,
 'num_units_lstm2': 499,
 'num_units_lstm3': 192,
 'dropout_rate': 0.3226826286678739,
 'learning_rate': 2.8689287823630512e-05}

# Función para obtener el modelo compilado
def create_model():
    # Entrenamiento de una nueva red neuronal con los mejores hiperparámetros
    best_model = Sequential(name='LSTM_Stack')
    best_model.add(LSTM(units=best_params['num_units_lstm1'], return_sequences=True, input_shape=(N_STEPS, 1), name='LSTM_L1'))
    best_model.add(Dropout(best_params['dropout_rate']))
    best_model.add(LSTM(units=best_params['num_units_lstm2'], return_sequences=True, name='LSTM_L2'))
    best_model.add(Dropout(best_params['dropout_rate']))
    best_model.add(LSTM(units=best_params['num_units_lstm3'], name='LSTM_L3'))
    best_model.add(Dense(units=20, name='Dense_L4'))
    best_model.add(Dense(units=1, name='Dense_L5'))
    best_model.add(Activation('linear'))

    optimizer = Adam(learning_rate=best_params['learning_rate'])
    best_model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])

    return best_model

# Crear el modelo compilado
model = create_model()

# Entrenamiento del modelo con los datos completos
history = model.fit(trainX, trainY,
            batch_size = BATCH_SIZE,
            epochs = 80,
            validation_split=0.05,
            callbacks=[early_stopping])

# Realizar la predicción
forecast = model.predict(testX) # notar esta predicción es con los datos de prueba y está con la escala de 0 a 1
forecast_prices = scaler.inverse_transform(forecast).squeeze()

def results_plotter(df_test=df_test, dataset_test=dataset_test, forecast=forecast):
    plt.figure(figsize=(16,10))
    dates = df_test.index[N_STEPS+1:,]
    plt.plot(dates, dataset_test[N_STEPS+1:,], label='BTC-USD real')
    plt.plot(dates, scaler.inverse_transform(forecast), label='BTC-USD forecast')
    plt.xlabel('date')
    plt.ylabel(f'{STOCK}')
    # Establecer intervalos de visualización en el eje x
    num_ticks = 8  # Número deseado de ticks en el eje x
    step = len(dates) // num_ticks  # Calcular el paso entre los ticks
    plt.xticks(dates[::step])  # Establecer los ticks en intervalos equidistantes
    plt.legend()
    plt.show()

# results_plotter()

# guardar los resultados en un DataFrame
dates = df_test.index[N_STEPS+1:,]
df_output = pd.DataFrame({ 'Date': dates, 'Forecast': forecast_prices})
df_output.head()

# Hacer un post a la API de AWS con los resultados
url = "https://1sxs2ownii.execute-api.us-east-1.amazonaws.com/qa/upload"
headers = {
    'Content-Type': 'application/json'
}

data = []
payload = {
        "data": data
    }
for index in df_output.index:
    row = df_output.iloc[index]
    fecha = row['Date']
    precio = row['Forecast']
    data.append({
            "id": str(index),
            "forecast": str(precio),
            "day": str(fecha),
            "currency": str(STOCK)
        })
response = requests.post(url, data=json.dumps(payload), headers=headers)