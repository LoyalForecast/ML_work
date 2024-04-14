import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
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

def create_model(optimizer='adam', neurons=32):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

nn_model = KerasClassifier(build_fn=create_model, verbose=0)

nn_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', nn_model)
])

param_grid = {
    'classifier__epochs': [50, 100],
    'classifier__batch_size': [10, 20, 50],
    'classifier__optimizer': ['Adam', 'Nadam'],
    'classifier__neurons': [16, 32, 64]
}

clf = RandomizedSearchCV(estimator=nn_pipeline, param_distributions=param_grid, n_iter=10, scoring='roc_auc', verbose=1, cv=3, n_jobs=-1)
search_result = clf.fit(X_train, y_train)

best_nn_model = search_result.best_estimator_

y_pred_nn = best_nn_model.predict(X_test)
y_pred_proba_nn = best_nn_model.predict_proba(X_test)[:, 1]
roc_auc_nn = roc_auc_score(y_test, y_pred_proba_nn)

print("Neural Network Model Classification Report:")
print(classification_report(y_test, y_pred_nn))
print("ROC AUC Score for Neural Network:", roc_auc_nn)

fpr_nn, tpr_nn, _ = roc_curve(y_test, y_pred_proba_nn)
plt.figure(figsize=(10, 5))
plt.plot(fpr_nn, tpr_nn, label='Neural Network (area = %0.2f)' % roc_auc_nn)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Neural Network')
plt.legend(loc="lower right")
plt.savefig('nn_roc_curve.png')
plt.show()

logging.info(f"Neural Network trained with ROC AUC: {roc_auc_nn}")

X = full_data.drop(['churn', 'npo_account_id', 'client_id', 'frst_pmnt_date', 'lst_pmnt_date_per_qrtr', 'year'], axis=1)
y = full_data['churn'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=3, subsample=0.75, verbosity=1, use_label_encoder=False, eval_metric='logloss'),
}

for name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"{name} Classifier Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score for {name}:", roc_auc)
    print(f"Precision-Recall AUC for {name}:", pr_auc)
    logging.info(f"{name} Model trained with ROC AUC: {roc_auc} and PR AUC: {pr_auc}")
