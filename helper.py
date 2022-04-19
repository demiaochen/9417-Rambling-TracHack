import pandas as pd
from datetime import datetime

def get_data():
    # get the data set
    df1 = pd.read_csv('../data/ebb_set1.csv')
    df2 = pd.read_csv('../data/ebb_set2.csv')
    X_df = pd.concat([df1, df2], ignore_index=True)
    X_df['ebb_eligible'] = X_df['ebb_eligible'].fillna(0)
    loyalty1 = pd.read_csv('../data/loyalty_program_ebb_set1.csv')
    loyalty2 = pd.read_csv('../data/loyalty_program_ebb_set2.csv')
    loyalty = pd.concat([loyalty1, loyalty2], ignore_index=True)

    # get royalty enrollment year
    enrolled_year = []
    for date in loyalty['date']:
        getted_date = datetime.strptime(date, '%Y-%m-%d')
        enrolled_year.append(getted_date.year)
    loyalty['enrolled_year'] = enrolled_year
    loyalty = loyalty.drop(columns=['date'])

    # merge ebb set and loyalty program
    # based on customer id, and fill 'N' to lrp_enrolled
    # if the customer has not enrolled
    X_df = pd.merge(X_df, loyalty, how='outer', on='customer_id')
    X_df = X_df[X_df['last_redemption_date'].notna()]
    X_df['lrp_enrolled'] = X_df['lrp_enrolled'].fillna('N')

    # separate data set into features and classes
    y_df = X_df['ebb_eligible']
    X_df = X_df.drop(columns=['ebb_eligible', 'customer_id'])

    # get last redemption
    last_redemption_year = []
    for date in X_df['last_redemption_date']:
        getted_date = datetime.strptime(date, '%Y-%m-%d')
        last_redemption_year.append(getted_date.year)
    X_df['last_redemption_year'] = last_redemption_year
    X_df = X_df.drop(columns=['last_redemption_date'])

    # get first activation
    first_activation_year = []
    for date in X_df['first_activation_date']:
        getted_date = datetime.strptime(date, '%Y-%m-%d')
        first_activation_year.append(getted_date.year)
    X_df['first_activation_year'] = first_activation_year
    X_df = X_df.drop(columns=['first_activation_date'])

    # if manufacturer is not SAMSUNG INC, APPLE, Apple Inc, LG INC
    # turn into other
    remain = ['SAMSUNG INC', 'APPLE', 'Apple Inc', 'LG INC']
    X_df['manufacturer'] = [x if x in remain else 'other' for x in X_df['manufacturer']]
    X_df['manufacturer'] = ['APPLE' if x == 'Apple Inc' else x for x in X_df['manufacturer']]

    # if manufacturer is not ios or android
    # turn into other
    remain.clear()
    remain = ['iOS', 'Android']
    X_df['operating_system'] = ['iOS' if x == 'IOS' else x for x in X_df['operating_system']]
    X_df['operating_system'] = ['Android' if x == 'ANDROID' else x for x in X_df['operating_system']]
    X_df['operating_system'] = [x if x in remain else 'other' for x in X_df['operating_system']]

    # fill 'None' if there is missing value
    X_df['language_preference'] = X_df['language_preference'].fillna('None')
    X_df['state'] = X_df['state'].fillna('None')
    
    # evaluate set
    evaluate = pd.read_csv('../data/eval_set.csv')
    customer_ids = evaluate['customer_id'].to_numpy()
    #X_df.drop(df.index, inplace=True)

    loyalty = pd.read_csv('../data/loyalty_program_eval_set.csv')
    enrolled_year.clear()
    for date in loyalty['date']:
        getted_date = datetime.strptime(date, '%Y-%m-%d')
        enrolled_year.append(getted_date.year)
    loyalty['enrolled_year'] = enrolled_year
    loyalty = loyalty.drop(columns=['date'])

    evaluate = pd.merge(evaluate, loyalty, how='outer', on='customer_id')
    evaluate = evaluate[evaluate['last_redemption_date'].notna()]
    evaluate['lrp_enrolled'] = evaluate['lrp_enrolled'].fillna('N')
    evaluate = evaluate.drop(columns=['customer_id'])

    # get last redemption
    last_redemption_year.clear()
    for date in evaluate['last_redemption_date']:
        getted_date = datetime.strptime(date, '%Y-%m-%d')
        last_redemption_year.append(getted_date.year)
    evaluate['last_redemption_year'] = last_redemption_year
    evaluate = evaluate.drop(columns=['last_redemption_date'])

    # get first activation
    first_activation_year.clear()
    for date in evaluate['first_activation_date']:
        getted_date = datetime.strptime(date, '%Y-%m-%d')
        first_activation_year.append(getted_date.year)
    evaluate['first_activation_year'] = first_activation_year
    evaluate = evaluate.drop(columns=['first_activation_date'])

    # if manufacturer is not SAMSUNG INC, APPLE, Apple Inc, LG INC
    # turn into other
    remain = ['SAMSUNG INC', 'APPLE', 'Apple Inc', 'LG INC']
    evaluate['manufacturer'] = [x if x in remain else 'other' for x in evaluate['manufacturer']]
    evaluate['manufacturer'] = ['APPLE' if x == 'Apple Inc' else x for x in evaluate['manufacturer']]

    # if manufacturer is not ios or android
    # turn into other
    remain.clear()
    remain = ['iOS', 'Android']
    evaluate['operating_system'] = ['iOS' if x == 'IOS' else x for x in evaluate['operating_system']]
    evaluate['operating_system'] = ['Android' if x == 'ANDROID' else x for x in evaluate['operating_system']]
    evaluate['operating_system'] = [x if x in remain else 'other' for x in evaluate['operating_system']]


    evaluate['language_preference'] = evaluate['language_preference'].fillna('None')
    evaluate['state'] = evaluate['state'].fillna('None')
    
    return X_df, y_df, evaluate, customer_ids