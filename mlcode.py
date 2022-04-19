import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from helper import get_data

# get data set and other information from helper
X_df, y_df, evaluate, customer_ids = get_data()

# train the model
cat_features = ['manufacturer', 'operating_system', 'language_preference', 'state', 'lrp_enrolled']
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.3, random_state=23)
cbc = CatBoostClassifier(iterations=2700, learning_rate=0.025, loss_function='Logloss')
model = cbc.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test), verbose=False, plot=True)

# evaluation for training
y_train_pred = model.predict(X_train)
train_score = f1_score(y_train, y_train_pred, average='weighted')
print(f'CatBoost train F1 score: {train_score}')
# evaluation for testing
y_test_pred = model.predict(X_test)
test_score = f1_score(y_test, y_test_pred, average='weighted')
print(f'Catboost test F1 score: {test_score}')

# write to csv
prediction = model.predict(evaluate)
result = pd.DataFrame({'customer_id': customer_ids, 'ebb_eligible':prediction})
result.to_csv('../submission/2022-04-17_final.csv', index=False)