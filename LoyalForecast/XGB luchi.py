import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import logging

logging.basicConfig(filename='model_logs.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)

def load_and_prepare_data(file_path, date_cols=None, dtype_dict=None, delimiter=','):
    data = pd.read_csv(file_path, delimiter=delimiter, parse_dates=date_cols, dtype=dtype_dict)
    logging.info(f"Loaded data from {file_path} with shape {data.shape}")
    return data

base_path = '/Users/daniil/Desktop/LoyalForecast (ЦП)/LoyalForecast/'
train_path = base_path + 'train.csv'
transactions_path = base_path + 'trnsctns.csv'
contributors_path = base_path + 'cntrbtrs.csv'

clnts_fin_prtrt = load_and_prepare_data(
    train_path,
    date_cols=['frst_pmnt_date', 'lst_pmnt_date_per_qrtr'],
    dtype_dict={'pmnts_type': 'category', 'gender': 'category'},
    delimiter=','
)

trnsctns = load_and_prepare_data(
    transactions_path,
    date_cols=['npo_oprtn_date'],
    dtype_dict={'npo_oprtn_grp': 'category'},
    delimiter=';'
)

cntrbtrs = load_and_prepare_data(
    contributors_path,
    delimiter=';',
    dtype_dict={'accnt_pnsn_schm': 'category'}
)

cntrbtrs.rename(columns={'npo_accnt_id': 'npo_account_id'}, inplace=True)

trnsctns['year_quarter'] = trnsctns['npo_oprtn_date'].dt.to_period('Q')
trns_aggregated = trnsctns.groupby('npo_account_id').agg({
    'npo_sum': ['sum', 'mean', 'count'],
    'npo_oprtn_grp': lambda x: x.mode()[0] if not x.empty else np.nan
}).reset_index()
trns_aggregated.columns = ['npo_account_id', 'total_sum', 'mean_sum', 'count_trns', 'common_oprtn_grp']

full_data = pd.merge(pd.merge(clnts_fin_prtrt, trns_aggregated, on='npo_account_id', how='left'), cntrbtrs, on='npo_account_id', how='left')

full_data['days_since_first_payment'] = (pd.Timestamp('today') - full_data['frst_pmnt_date']).dt.days
full_data['days_since_last_payment'] = (pd.Timestamp('today') - full_data['lst_pmnt_date_per_qrtr']).dt.days

features = ['age', 'clnt_cprtn_time_d', 'actv_prd_d', 'lst_pmnt_rcnc_d', 'balance', 'oprtn_sum_per_qrtr',
            'oprtn_sum_per_year', 'pmnts_sum', 'pmnts_nmbr', 'pmnts_sum_per_qrtr', 'pmnts_sum_per_year',
            'pmnts_nmbr_per_qrtr', 'pmnts_nmbr_per_year', 'incm_sum', 'incm_per_qrtr', 'incm_per_year',
            'mgd_accum_period', 'mgd_payment_period', 'days_since_first_payment', 'days_since_last_payment']
categorical_features = ['pmnts_type', 'gender', 'common_oprtn_grp', 'accnt_pnsn_schm', 'quarter', 'assignee_npo', 
                        'assignee_ops', 'phone_number', 'email', 'lk', 'citizen', 'fact_addrss', 'appl_mrkr', 'evry_qrtr_pmnt']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=100, max_depth=3, subsample=0.75, verbosity=1, use_label_encoder=False, eval_metric='logloss'))
])

X = full_data.drop(['churn', 'npo_account_id', 'client_id', 'frst_pmnt_date', 'lst_pmnt_date_per_qrtr', 'year'], axis=1)
y = full_data['churn'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("XGBoost Classifier Report:")
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC AUC Score:", roc_auc)

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)
print("Precision-Recall AUC:", pr_auc)

plt.figure(figsize=(10, 5))
plt.plot(recall, precision, label='PR Curve (area = %0.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="upper right")
plt.savefig('pr_curve.png')
plt.show()

logging.info(f"Model trained with ROC AUC: {roc_auc} and PR AUC: {pr_auc}")

predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Predicted_Proba': y_pred_proba})
predictions.to_csv('model_predictions.csv', index=False)