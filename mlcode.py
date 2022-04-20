import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from helper import get_data
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np

# get data set and other information from helper
X_df, y_df, evaluate, customer_ids = get_data()

# plot
keys = []
values = []
categories = ['manufacturer', 'operating_system', 'language_preference', 'state']
for feature in categories:
    data = X_df[feature].value_counts()
    keys.clear()
    values.clear()
    for key, value in data.items():
        keys.append(key)
        values.append(value / X_df.shape[0])
    
    keys = keys[0:5]
    values = values[0:5]
    nan_data = {'features': keys, 'percentage': values}
    plt.figure(figsize=(45, 20))
    sbn.barplot(x=nan_data['features'], y=nan_data['percentage'])
    plt.xlabel(feature)
    plt.ylabel('Percentage')
    plt.title('Top 5 Percentage of categories')
    plt.show()

nan_data = X_df.isna().sum()
keys.clear()
values.clear()
for key, value in nan_data.items():
    if value != 0:
        keys.append(key)
        values.append(value / X_df.shape[0] * 100)

nan_data = {'features': keys, 'percentage': values}
plt.figure(figsize=(45, 20))
sbn.barplot(x=nan_data['features'], y=nan_data['percentage'])

plt.xlabel('Features')
plt.ylabel('Percentage of NaN(%)')
plt.title('Percentage of NaN of Features with NaN')
plt.show()

# train the model
cat_features = ['manufacturer', 'operating_system', 'language_preference', 'state', 'lrp_enrolled']
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.3, random_state=23)
cbc = CatBoostClassifier(iterations=2700, learning_rate=0.025, loss_function='Logloss')
model = cbc.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test), verbose=False, plot=True)

# plot the importance
importance = model.get_feature_importance()
features = np.array(X_train.columns)
data = {'features_name': features, 'features_importance': importance}
data = pd.DataFrame(data)

# sort the importance
data.sort_values(by=['features_importance'], ascending=False, inplace=True)

plt.figure(figsize=(45, 20))
sbn.barplot(x=data['features_name'], y=data['features_importance'])

plt.title('Features Importance Plot')
plt.xlabel('Features Name')
plt.ylabel('Features Importance')
plt.show

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