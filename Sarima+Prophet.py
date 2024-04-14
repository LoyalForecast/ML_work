import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fbprophet import Prophet
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_data(filepath, index_col, parse_dates=True):
    return pd.read_csv(filepath, index_col=index_col, parse_dates=parse_dates)

base_path = '/Users/daniil/Desktop/LoyalForecast (ЦП)/ML_work/ML_work/'
transactions_path = base_path + 'trnsctns.csv'

data = load_data(transactions_path, index_col='npo_oprtn_date')

data = data.resample('M').sum()

def sarima_forecast(history):
    model = SARIMAX(history, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]

def prophet_forecast(data):
    df = pd.DataFrame(data.index)
    df.columns = ['ds']
    df['y'] = data.values
    prophet = Prophet(yearly_seasonality=True, daily_seasonality=False)
    prophet.fit(df)
    future = prophet.make_future_dataframe(periods=1, freq='M')
    forecast = prophet.predict(future)
    yhat = forecast['yhat'].iloc[-1]
    return yhat

train, test = data.iloc[:-12], data.iloc[-12:]

predictions_sarima = list()
for t in range(len(test)):
    yhat = sarima_forecast(train['npo_sum'])
    predictions_sarima.append(yhat)
    train = train.append(test.iloc[t])

predictions_prophet = list()
yhat = prophet_forecast(train['npo_sum'])
predictions_prophet.append(yhat)

mse_sarima = mean_squared_error(test['npo_sum'], predictions_sarima)
mse_prophet = mean_squared_error(test['npo_sum'], [predictions_prophet[-1]] * len(test))

print('SARIMA MSE:', mse_sarima)
print('Prophet MSE:', mse_prophet)

plt.figure(figsize=(12, 6))
plt.plot(test.index, test['npo_sum'], label='Actual')
plt.plot(test.index, predictions_sarima, color='red', label='SARIMA')
plt.plot(test.index, [predictions_prophet[-1]] * len(test), color='green', label='Prophet')
plt.title('Forecast vs Actuals')
plt.legend()
plt.show()
