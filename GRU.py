import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(filename='deep_learning_time_series.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

def load_data(filepath, parse_dates, dtype, sep):
    data = pd.read_csv(filepath, sep=sep, parse_dates=parse_dates, dtype=dtype)
    logging.info(f"Loaded data from {filepath} with shape {data.shape}")
    return data

base_path = '/Users/daniil/Desktop/LoyalForecast (ЦП)/ML_work/ML_work/'
train_path = f'{base_path}train.csv'
transactions_path = f'{base_path}trnsctns.csv'
contributors_path = f'{base_path}cntrbtrs.csv'

data = load_data(train_path, ['date_column'], {'category_feature': 'category'}, ',')

data['year'] = data['date_column'].dt.year
data['month'] = data['date_column'].dt.month
data['day'] = data['date_column'].dt.day

features = ['year', 'month', 'day'] + [col for col in data.columns if col.startswith('feature')]
target = 'target_column'

data.sort_values('date_column', inplace=True)
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_data[features])
test_features = scaler.transform(test_data[features])

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 30
X_train, y_train = create_dataset(train_features, train_data[target].values, time_steps)
X_test, y_test = create_dataset(test_features, test_data[target].values, time_steps)

model = Sequential([
    LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')


history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
logging.info(f"Model MSE: {mse}, R2 Score: {r2}")

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True')
plt.plot(predictions, label='Predicted')
plt.title('LSTM Forecast vs Actuals')
plt.legend()
plt.show()

model.save('lstm_model.h5')
logging.info("Model trained and saved successfully.")

