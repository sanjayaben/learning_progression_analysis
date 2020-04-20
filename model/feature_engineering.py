import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


def list_null_values_original_columns(analysis_with_history_file):
    '''
      INPUT
      analysis_with_history_file - the pickle file with expanded KC and success details

      OUTPUT
      None

      Analysis of the null values in the original 19 columns
      '''
    df = pd.read_pickle(analysis_with_history_file)
    df_original = df.iloc[:,:19]
    print(df_original.isnull().sum())

def drop_unnessasary_columns(analysis_with_history_file, refined_data_file):
    '''
      INPUT
      analysis_with_history_file - the pickle file with expanded KC and success details
      refined_data_file - the output data file

      OUTPUT
      None

      drop columns not usefull for modelling
      '''
    df = pd.read_pickle(analysis_with_history_file)
    df = df.drop(columns=['Step Start Time', 'First Transaction Time','Correct Transaction Time','Step End Time', 'Correct Step Duration (sec)','Error Step Duration (sec)','KC(Default)','Opportunity(Default)' ])
    df.to_pickle(refined_data_file)

def fix_null_values(refined_data_file):
    '''
      INPUT
      refined_data_file - the output data file

      OUTPUT
      None

      fix null values by dropping small number of rows in 'Step Duration (sec)' and fill the rest with 0
      '''
    df = pd.read_pickle(refined_data_file)
    #drop Step Duration (sec) - 919 null records
    print('before==>', df.shape)
    df = df.dropna(subset=['Step Duration (sec)'])
    print('after==>', df.shape)
    print(df.iloc[:, :11].isnull().sum())
    #fill other null values with 0
    df = df.fillna(0)
    print(df.isnull().sum())
    df.to_pickle(refined_data_file)

def scale_features(refined_data_file, training_data_file):
    '''
      INPUT
      refined_data_file - the input data file
      training_data_file - the file saving the output data

      OUTPUT
      None

      scale numerical values to range 0-1 using the MinMaxScaler
      '''
    df = pd.read_pickle(refined_data_file)
    min_max_scaler = MinMaxScaler()
    df.iloc[:, 6:] = min_max_scaler.fit_transform(df.iloc[:, 6:])
    df.to_pickle(training_data_file)

def add_agg_features(training_data_file,training_data_final_file):
    '''
      INPUT
      training_data_file - the input data file
      training_data_final_file - the file saving the output data

      OUTPUT
      None

      generate some new features considering the average correct first attempt success rate.
      The new features represent the difficulty at problem hierarchy, problem and step levels
      '''
    df = pd.read_pickle(training_data_file)
    df['difficulty_PH'] = df.groupby('Problem Hierarchy')['Correct First Attempt'].transform(np.mean)
    df['difficulty_Prob'] = df.groupby('Problem Name')['Correct First Attempt'].transform(np.mean)
    df['difficulty_Step'] = df.groupby(['Problem Name','Step Name'])['Correct First Attempt'].transform(np.mean)

    df = df.drop(columns=['Row', 'Anon Student Id','Problem Hierarchy','Problem Name', 'Problem View','Step Name'])
    df.to_pickle(training_data_final_file)

def print_correlation_matrix(training_data_final_file):
    '''
      INPUT
      training_data_final_file - final dataset

      OUTPUT
      None

      Shows the correlation matrix for the numerical features except the expanded KC related ones
      Idea is to findout whether we should drop further features
      '''
    df = pd.read_pickle(training_data_final_file)
    df_first_cols = df[['Step Duration (sec)','Correct First Attempt','Incorrects','Hints','Corrects','difficulty_PH','difficulty_Prob','difficulty_Step']]
    sns.set()
    plt.figure(figsize=(10, 5))
    sns.heatmap(df_first_cols.corr(), annot=True, linewidths=.5)
    plt.show()


def main():

    analysis_with_history_file = '../data/output/analysis_with_history_df.pickle'
    refined_data_file = "../data/output/refined_data.pickle"
    training_data_file = '../data/output/training_data.pickle'
    training_data_final_file = '../data/output/training_data_final.pickle'

    list_null_values_original_columns(analysis_with_history_file)
    drop_unnessasary_columns(analysis_with_history_file,refined_data_file)
    fix_null_values(refined_data_file)
    scale_features(refined_data_file,training_data_file)
    add_agg_features(training_data_file,training_data_final_file)
    print_correlation_matrix(training_data_final_file)

if __name__ == '__main__':
    main()