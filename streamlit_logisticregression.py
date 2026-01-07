import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 1. 数据处理函数
def get_my_data():
    df = pd.read_csv("data.csv")
    # 修正：必须赋值
    df = df.drop('ID', axis=1) 

    df['km_per_driving_day'] = df['driven_km_drives'] / df['driving_days']
    df.loc[df['km_per_driving_day'] == np.inf, 'km_per_driving_day'] = 0

    df['professional_driver'] = np.where((df['drives'] >= 60) & (df['driving_days'] >= 15), 1, 0)
    df = df.dropna(subset=['label'])

    # 异常值处理
    for column in ['sessions', 'drives', 'total_sessions', 'total_navigations_fav1',
               'total_navigations_fav2', 'driven_km_drives', 'duration_minutes_drives']:
        threshold = df[column].quantile(0.95)
        df.loc[df[column] > threshold, column] = threshold

    df['label2'] = np.where(df['label'] == 'churned', 1, 0)
    return df

df = get_my_data()

# 2. Logistic Regression
def train_save_model_LR(df):
    X = df.drop(columns=['label', 'label2', 'device', 'sessions', 'driving_days'])
    y = df['label2']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    clf = LogisticRegression(penalty='none', max_iter=500)
    clf.fit(X_train, y_train)
    # save model
    joblib.dump(clf, 'lr_model.pkl')
    return clf, X_test, y_test

# 3. Random Forest
def train_save_model_rf(df):
    X = df.drop(columns=['label', 'label2', 'device'])
    y = df['label2']
    # split data
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=42)
    
    rf = RandomForestClassifier(random_state=42)
    cv_params = {'max_depth': [None], 'n_estimators': [300], 'min_samples_leaf': [2]}
    rf_cv = GridSearchCV(rf, cv_params, scoring='recall', cv=4)
    rf_cv.fit(X_train, y_train)
    
    best_rf = rf_cv.best_estimator_
    # save best_rf
    joblib.dump(best_rf, 'rf_model.pkl') 
    return best_rf, X_test, y_test

# 4. XGBoost
def train_save_model_xgb(df):
    X = df.drop(columns=['label', 'label2', 'device'])
    y = df['label2']
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=42)
    
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_cv = GridSearchCV(xgb, {'max_depth': [5], 'n_estimators': [300]}, scoring='recall', cv=4)
    xgb_cv.fit(X_train, y_train)
    
    best_xgb = xgb_cv.best_estimator_
    # save best_xgb
    joblib.dump(best_xgb, 'xgb_model.pkl')
    return best_xgb, X_test, y_test

# save the test data for Streamlit display
clf, _, _ = train_save_model_LR(df)
rf_model, _, _ = train_save_model_rf(df)
xgb_model, X_test, y_test = train_save_model_xgb(df)

# 为了 Streamlit 展示，建议把测试集也存一下
X_test.to_csv("test_x.csv", index=False)
y_test.to_csv("test_y.csv", index=False)