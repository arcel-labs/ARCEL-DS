import numpy as np #linear algebra
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from flask import request
import pickle
sns.set(style='ticks')


def import_dataframe():
    df_mat = pd.read_csv('student-mat.csv', sep=';', encoding='utf-8')
    df_por = pd.read_csv('student-por.csv', sep=';', encoding='utf-8')
  
    return pd.concat([df_mat, df_por])

def create_nota_total(df): 
    df['total_grade'] = (df['G1']+df['G2']+df['G3'])/3
    df = df.drop(columns=['G1','G2','G3', 'famsup', 'schoolsup','paid', 'activities', 'nursery', 'romantic', 'goout', 'Dalc', 'Walc', 'health', 'sex', 'Mjob', 'Fjob', 'reason', 'address'])

    return df

def rename_columns(df):
    df.columns = df.columns.str.capitalize()
    df.rename(columns={'Famsize': 'family_size',
                   'Pstatus': 'parent_status',
                   'Medu': 'mother_education',
                   'Fedu': 'father_Education',
                   'Traveltime': 'travel_time',
                   'Studytime': 'study_time',
                   'Internet': 'internet_access',
                   'Famrel': 'family_relationship',
                   'Freetime': 'free_time',
                   'Absences': 'absences',
                   'total_grade': 'total_grade'}, inplace=True)
    return df

def correlation_matrix(df):

    plt.figure(figsize=(17, 15))
    corr_mask = np.triu(df.corr())
    h_map = sns.heatmap(df.corr(), mask=corr_mask, annot=True, cmap='Blues')
    plt.yticks(rotation=360)
    plt.xticks(rotation=90)
    plt.savefig('heatmap.png')

def dashboards(df):
    sns.catplot(x='School', hue='Failures', col='Age', data=df, kind='count', palette='flare')
    plt.savefig('failure_compare')

#Label Conversion on The Dataset
def label_conversion(df): 

    df['family_size'] =  df['family_size'].apply(lambda x:x.replace('LE3', '1').replace('GT3', '2')).astype(int)
    df['parent_status'] =  df['parent_status'].apply(lambda x:x.replace('T', '1').replace('A', '2')).astype(int)
    df['internet_access'] =  df['internet_access'].apply(lambda x:x.replace('yes', '1').replace('no', '0')).astype(int)
    df['Guardian'] = df['Guardian'].apply(lambda x:x.replace('mother', '1').replace('father', '2').replace('other', '3')).astype(int)
    df['Higher'] = df['Higher'].apply(lambda x:x.replace('yes', '1').replace('no', '0')).astype(int)
    df['School'] = df['School'].apply(lambda x:x.replace('GP', '1').replace('MS', '2') ).astype(int)
    

    df.columns= df.columns.str.lower()

    return df

def feature_separating(df):
    X = df.drop(columns = ['total_grade'])
    y = df['total_grade'].astype(int)
    return X, y

def modeling(df, X, y):
    #Preparing Training, Testing, And Validating Dataset
    from sklearn.model_selection import train_test_split
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

    from sklearn.ensemble import GradientBoostingRegressor
    model_boost = GradientBoostingRegressor()
    model_boost = model_boost.fit(X_train, y_train)
#save model
    filename = 'boost_model.sav'
    pickle.dump(model_boost, open(filename, 'wb'))

    y_pred_rfr = model_boost.predict(X_test)

    #Evaluating The Machine Learning Model
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    #Mean Squared Error
    mse = mean_squared_error(y_test, y_pred_rfr)

    #Mean Absolute Error
    mae = mean_absolute_error(y_test, y_pred_rfr)

    #Root Mean Squared Error
    rmse = np.sqrt(mse)


    fig = plt.figure(figsize=(17, 10))
    df = df.sort_values(by=['total_grade'])
    X = df.drop('total_grade', axis=1)
    y = df['total_grade']
    plt.scatter(range(X.shape[0]), model_boost.predict(X), marker='.', label='Predict')
    plt.scatter(range(X.shape[0]), y, color='red', label='Real')
    plt.legend(loc='best', prop={'size': 10})
    plt.savefig('validate_model')
    
    return mse, mae, rmse


def main():

    df = import_dataframe()
    df = create_nota_total(df)
    #print(df)
    df = rename_columns(df)
    correlation_matrix(df)
    dashboards(df)
    
    df = label_conversion(df)
    X, y = feature_separating(df)
    print(df.columns)
    mse, mae, rmse = modeling(df, X, y)

    df['idUser'] = df.index + 1
    df.to_csv('alunos.csv')

main()
