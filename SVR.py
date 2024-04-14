import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(filename='model_performance.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

def load_data(filepath, parse_dates, dtype, sep):
    return pd.read_csv(filepath, sep=sep, parse_dates=parse_dates, dtype=dtype)

base_path = '/Users/daniil/Desktop/LoyalForecast (ЦП)/ML_work/ML_work/'
train_path = f'{base_path}train.csv'
transactions_path = f'{base_path}trnsctns.csv'
contributors_path = f'{base_path}cntrbtrs.csv'

clnts_fin_prtrt = load_data(train_path, ['frst_pmnt_date', 'lst_pmnt_date_per_qrtr'], {'pmnts_type': 'category', 'gender': 'category'}, ',')
transactions = load_data(transactions_path, ['npo_oprtn_date'], {'npo_oprtn_grp': 'category'}, ';')
contributors = load_data(contributors_path, None, {'accnt_pnsn_schm': 'category'}, ';')

contributors.rename(columns={'npo_accnt_id': 'npo_account_id'}, inplace=True)
transactions['year_quarter'] = transactions['npo_oprtn_date'].dt.to_period('Q')
agg_funcs = {'npo_sum': ['sum', 'mean', 'count'], 'npo_oprtn_grp': lambda x: x.mode()[0] if not x.empty else np.nan}
transactions_agg = transactions.groupby('npo_account_id').agg(agg_funcs).reset_index()
transactions_agg.columns = ['npo_account_id', 'total_sum', 'mean_sum', 'count_trns', 'common_oprtn_grp']
full_data = pd.merge(pd.merge(clnts_fin_prtrt, transactions_agg, on='npo_account_id'), contributors, on='npo_account_id')
full_data['days_since_first_payment'] = (pd.Timestamp('today') - full_data['frst_pmnt_date']).dt.days
full_data['days_since_last_payment'] = (pd.Timestamp('today') - full_data['lst_pmnt_date_per_qrtr']).dt.days

features = ['age', 'balance', 'total_sum', 'mean_sum', 'count_trns', 'days_since_first_payment', 'days_since_last_payment']
categorical_features = ['pmnts_type', 'gender', 'common_oprtn_grp', 'accnt_pnsn_schm', 'quarter', 'assignee_npo', 'assignee_ops', 'phone_number', 'email', 'lk', 'citizen', 'fact_addrss', 'appl_mrkr', 'evry_qrtr_pmnt']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('imputer', SimpleImputer(strategy='mean'), features)
    ])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('svr', SVR(C=1.0, epsilon=0.2))
])

X = full_data[features + categorical_features]
y = full_data['balance']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions.to_csv(f'{base_path}svr_predictions.csv', index=False)
plt.figure(figsize=(10, 6))
sns.regplot(x='Actual', y='Predicted', data=predictions, fit_reg=True, scatter_kws={"s": 100})
plt.xlabel('Actual Balance')
plt.ylabel('Predicted Balance')
plt.title('SVR Predictions vs Actual')
plt.savefig(f'{base_path}svr_plot.png')
plt.show()

logging.info(f"SVR Model - MSE: {mse}, R^2: {r2}")

