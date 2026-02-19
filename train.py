import numpy as no
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
df = pd.read_csv('global_water_consumption_2000_2025.csv')


label_map = {
    'Low' : 0,
    'Moderate' : 1,
    'High' : 2,
    'Critical' : 3
}

df['Water Scarcity Level'] = df['Water Scarcity Level'].map(label_map)

X = df.drop(columns='Water Scarcity Level')
y = df['Water Scarcity Level']

num_features = X.select_dtypes(include=['float64', 'int64']).columns
cat_features = X.select_dtypes(include=['object']).columns

num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', cat_transformer, cat_features),
        ('num', num_transformer, num_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = XGBClassifier(
    objective='multi:softmax',
    num_class=4,
    n_estimators=300,
    max_depth=None,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])

model_pipe.fit(X_train, y_train)

with open('water.pkl', 'wb') as f:
    pickle.dump(model_pipe,f)