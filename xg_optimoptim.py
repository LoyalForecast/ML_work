import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

def load_and_prepare_data(file_path, date_cols, dtype_dict, delimiter):
    start_time = time.time()
    data = pd.read_csv(file_path, delimiter=delimiter, parse_dates=date_cols, dtype=dtype_dict)
    duration = time.time() - start_time
    print(f"Loaded data from {file_path} with shape {data.shape} in {duration:.2f} seconds")
    return data

base_path = '/Users/daniil/Desktop/LoyalForecast (ЦП)/ML_work/ML_work/'
train_path = base_path + 'train.csv'
transactions_path = base_path + 'trnsctns.csv'
contributors_path = base_path + 'cntrbtrs.csv'
test_path = base_path + 'test.csv'

clnts_fin_prtrt = load_and_prepare_data(train_path, ['frst_pmnt_date', 'lst_pmnt_date_per_qrtr'], {'pmnts_type': 'category', 'gender': 'category'}, ',')
trnsctns = load_and_prepare_data(transactions_path, ['npo_oprtn_date'], {'npo_oprtn_grp': 'category'}, ';')
cntrbtrs = load_and_prepare_data(contributors_path, None, {'accnt_pnsn_schm': 'category'}, ';')

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
categorical_features = ['pmnts_type', 'gender', 'quarter', 'accnt_pnsn_schm']  # Removed 'common_oprtn_grp'

num_preprocessor = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_preprocessor = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', num_preprocessor, features), ('cat', cat_preprocessor, categorical_features)])

model = Pipeline([('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=100))])

X = full_data.drop(['churn', 'npo_account_id', 'client_id', 'frst_pmnt_date', 'lst_pmnt_date_per_qrtr', 'year'], axis=1)
y = full_data['churn'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

start_train = time.time()
model.fit(X_train, y_train)
train_duration = time.time() - start_train
print(f"Training completed in {train_duration:.2f} seconds")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]
print("Logistic Regression Classifier Report:")
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

test_data = pd.read_csv(test_path)
test_data['days_since_first_payment'] = (pd.Timestamp('today') - pd.to_datetime(test_data['frst_pmnt_date'])).dt.days
test_data['days_since_last_payment'] = (pd.Timestamp('today') - pd.to_datetime(test_data['lst_pmnt_date_per_qrtr'])).dt.days

# Handle missing categorical columns in the test data
missing_columns = set(categorical_features) - set(test_data.columns)
for col in missing_columns:
    test_data[col] = np.nan

test_data_prepared = preprocessor.transform(test_data.drop(['npo_account_id'], axis=1))
test_predictions = model.predict_proba(test_data_prepared)[:, 1]

submission = pd.DataFrame({
    'npo_account_id': test_data['npo_account_id'],
    'churn': test_predictions
})
submission.to_csv('submission.csv', index=False)