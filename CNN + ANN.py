import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

def load_data(file_path, date_cols, dtype_dict, delimiter):
    return pd.read_csv(file_path, delimiter=delimiter, parse_dates=date_cols, dtype=dtype_dict)

base_path = 'A:/WORK/цп/train/'
train_path = base_path + 'train.csv'
transactions_path = base_path + 'trnsctns.csv'
contributors_path = base_path + 'cntrbtrs.csv'

clnts_fin_prtrt = load_data(train_path, ['frst_pmnt_date', 'lst_pmnt_date_per_qrtr'], {'pmnts_type': 'category', 'gender': 'category'}, ',')
trnsctns = load_data(transactions_path, ['npo_oprtn_date'], {'npo_oprtn_grp': 'category'}, ';')
cntrbtrs = load_data(contributors_path, None, {'accnt_pnsn_schm': 'category'}, ';')

cntrbtrs.rename(columns={'npo_accnt_id': 'npo_account_id'}, inplace=True)
trnsctns['year_quarter'] = trnsctns['npo_oprtn_date'].dt.to_period('Q')
trns_aggregated = trnsctns.groupby('npo_account_id').agg({'npo_sum': ['sum', 'mean', 'count'], 'npo_oprtn_grp': lambda x: x.mode()[0] if not x.empty else np.nan}).reset_index()
trns_aggregated.columns = ['npo_account_id', 'total_sum', 'mean_sum', 'count_trns', 'common_oprtn_grp']
full_data = pd.merge(pd.merge(clnts_fin_prtrt, trns_aggregated, on='npo_account_id', how='left'), cntrbtrs, on='npo_account_id', how='left')
full_data['days_since_first_payment'] = (pd.Timestamp('today') - full_data['frst_pmnt_date']).dt.days
full_data['days_since_last_payment'] = (pd.Timestamp('today') - full_data['lst_pmnt_date_per_qrtr']).dt.days

features = ['age', 'clnt_cprtn_time_d', 'actv_prd_d', 'lst_pmnt_rcnc_d', 'balance', 'oprtn_sum_per_qrtr', 'oprtn_sum_per_year', 'pmnts_sum', 'pmnts_nmbr', 'pmnts_sum_per_qrtr', 'pmnts_sum_per_year', 'pmnts_nmbr_per_qrtr', 'pmnts_nmbr_per_year', 'incm_sum', 'incm_per_qrtr', 'incm_per_year', 'mgd_accum_period', 'mgd_payment_period', 'days_since_first_payment', 'days_since_last_payment']
categorical_features = ['pmnts_type', 'gender', 'common_oprtn_grp', 'accnt_pnsn_schm', 'quarter', 'assignee_npo', 'assignee_ops', 'phone_number', 'email', 'lk', 'citizen', 'fact_addrss', 'appl_mrkr', 'evry_qrtr_pmnt']

preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), features), ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])
X = full_data.drop(['churn', 'npo_account_id', 'client_id', 'frst_pmnt_date', 'lst_pmnt_date_per_qrtr', 'year'], axis=1)
y = full_data['churn'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

def create_ann_model():
    model = Sequential([Dense(64, input_dim=58, activation='relu'), Dropout(0.5), Dense(32, activation='relu'), Dropout(0.5), Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_cnn_model():
    model = Sequential([Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(58, 1)), MaxPooling1D(pool_size=2), Flatten(), Dense(50, activation='relu'), Dense(1, activation='sigmoid')])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

ann_model = create_ann_model()
ann_model.fit(X_train_preprocessed, y_train, epochs=50, batch_size=32, verbose=1)
y_pred_proba_ann = ann_model.predict(X_test_preprocessed).ravel()
roc_auc_ann = roc_auc_score(y_test, y_pred_proba_ann)

X_train_cnn = X_train_preprocessed.reshape(X_train_preprocessed.shape[0], X_train_preprocessed.shape[1], 1)
X_test_cnn = X_test_preprocessed.reshape(X_test_preprocessed.shape[0], X_test_preprocessed.shape[1], 1)
cnn_model = create_cnn_model()
cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, verbose=1)
y_pred_proba_cnn = cnn_model.predict(X_test_cnn).ravel()
roc_auc_cnn = roc_auc_score(y_test, y_pred_proba_cnn)

fpr_ann, tpr_ann, _ = roc_curve(y_test, y_pred_proba_ann)
fpr_cnn, tpr_cnn, _ = roc_curve(y_test, y_pred_proba_cnn)
plt.figure(figsize=(10, 5))
plt.plot(fpr_ann, tpr_ann, label=f'ANN (AUC = {roc_auc_ann:.2f})')
plt.plot(fpr_cnn, tpr_cnn, label=f'CNN (AUC = {roc_auc_cnn:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.show()