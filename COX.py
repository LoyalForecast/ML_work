import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lifelines import CoxPHFitter
from lifelines.utils import datetimes_to_durations
import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename='cox_model_logs.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

def load_and_prepare_data(file_path, date_cols=None, dtype_dict=None, delimiter=','):
    data = pd.read_csv(file_path, delimiter=delimiter, parse_dates=date_cols, dtype=dtype_dict)
    logging.info(f"Loaded data from {file_path} with shape {data.shape}")
    return data

base_path = '/Users/daniil/Desktop/LoyalForecast (ЦП)/ML_work/ML_work/'
train_path = f'{base_path}train.csv'
transactions_path = f'{base_path}trnsctns.csv'
contributors_path = f'{base_path}cntrbtrs.csv'

clnts_fin_prtrt = load_and_prepare_data(train_path, ['frst_pmnt_date', 'lst_pmnt_date_per_qrtr'], {'pmnts_type': 'category', 'gender': 'category'}, ',')
transactions = load_and_prepare_data(transactions_path, ['npo_oprtn_date'], {'npo_oprtn_grp': 'category'}, ';')
contributors = load_and_prepare_data(contributors_path, None, {'accnt_pnsn_schm': 'category'}, ';')

contributors.rename(columns={'npo_accnt_id': 'npo_account_id'}, inplace=True)
transactions['year_quarter'] = transactions['npo_oprtn_date'].dt.to_period('Q')
transactions_agg = transactions.groupby('npo_account_id').agg({'npo_sum': ['sum', 'mean', 'count'], 'npo_oprtn_grp': lambda x: x.mode()[0] if not x.empty else np.nan}).reset_index()
transactions_agg.columns = ['npo_account_id', 'total_sum', 'mean_sum', 'count_trns', 'common_oprtn_grp']

full_data = pd.merge(pd.merge(clnts_fin_prtrt, transactions_agg, on='npo_account_id', how='left'), contributors, on='npo_account_id', how='left')
full_data['days_since_first_payment'] = (pd.Timestamp('today') - full_data['frst_pmnt_date']).dt.days
full_data['days_since_last_payment'] = (pd.Timestamp('today') - full_data['lst_pmnt_date_per_qrtr']).dt.days

full_data['duration'], full_data['event_observed'] = datetimes_to_durations(full_data['frst_pmnt_date'], full_data['lst_pmnt_date_per_qrtr'], 'days')

covariates = ['age', 'total_sum', 'mean_sum', 'count_trns', 'days_since_first_payment', 'days_since_last_payment'] + pd.get_dummies(full_data[['gender', 'common_oprtn_grp', 'accnt_pnsn_schm']], drop_first=True).columns.tolist()
cph = CoxPHFitter()
cph.fit(full_data[['duration', 'event_observed'] + covariates], duration_col='duration', event_col='event_observed')

print(cph.summary)

cph.plot_partial_effects_on_outcome(covariates=['age', 'total_sum'], values=[[50, 1000], [60, 2000]], cmap='coolwarm')
plt.title('Survival Function')
plt.savefig('survival_function.png')
plt.show()

logging.info("Cox Model Coefficients:\n" + str(cph.summary))
cph.summary.to_csv('cox_model_summary.csv')

predictions = cph.predict_partial_hazard(full_data[['duration', 'event_observed'] + covariates])
predictions.to_csv('cox_model_predictions.csv', index=False)
