import pandas as pd


def overview_data(training_data_file):
    '''
    INPUT
    training_data_file - The text/csv file including the dataset

    OUTPUT
    None

    Prints out some important information about the dataset
    '''
    df = pd.read_csv(training_data_file, sep="\t")
    print('Dataset columns => ', df.columns)
    print('Shape of the dataset => ', df.shape)
    print('Total number of unique students => ', len(set(df['Anon Student Id'])))
    print('Total number of unique problems => ', len(set(df['Problem Name'])))
    print("Descriptive Statistics => ", df.describe())


def expand_KC_with_history(training_data_file):
    '''
    INPUT
    training_data_file - The text/csv file including the dataset

    OUTPUT
    None

    This function is to expand the Knowledge Component (KC) information contained in the KC(Default) column. The function does few things
    1. Expand KC information into columns. This would mean that every KC value will be represented as a column in the dataframe and if the particular KC is
    applicable to the transaction row, then it would be represented with 1 (otherwise null which would later be set to 0).
    A side benefit is that it transform the categorical KC values in to a numerical representation.

    2. For each KC present in a transaction, we calculate the average success rate considering all previous transactions for the particular student,
    with this KC present. This acts as an accumulated success rate for the particular student/KC combination.
    '''
    df = pd.read_csv(training_data_file, sep="\t")
    #Sort by student id and transaction time. This enables efficincy when looping through student sets
    df = df.sort_values(by=['Anon Student Id', 'First Transaction Time'], ascending=True)
    df = df.reset_index(drop=True)
    #Multiple knowledge components are present seperated by ~~
    df_kc_vals = df['KC(Default)'].str.split("~~")

    row_index = 0
    prev_user_id = ''
    cur_start_index = 0
    for row in df_kc_vals:
        cur_user_id = df.at[row_index, 'Anon Student Id']
        #Change in student id means the begining of a new student record set
        if cur_user_id != prev_user_id:
            prev_user_id = cur_user_id
            cur_start_index = row_index
            print(cur_start_index, ' : ', row_index)

        if not str(row) == 'nan':
            inter_index = 0

            for row_val in row:
                repl_value = row_val.replace('[SkillRule: ', '')
                col_vals = repl_value.split(";")
                col = col_vals[0]
                df.at[row_index, col] = 1
                #Get all previous student records up to the current record with the KC present
                df_prev = df.iloc[cur_start_index:row_index]
                current_succ_rate = df_prev[(df_prev[col] == 1)].groupby([col])['Correct First Attempt'].mean() #Our success indicator is the correct first attempt
                if len(current_succ_rate.values) > 0:
                    df.at[row_index, 'SUC_' + col] = current_succ_rate.values[0]
                inter_index += 1
        row_index += 1
    df.to_pickle('output/analysis_with_history_df.pickle')
    print('++++++++++++ Finished +++++++++++++++++')



def create_correlation_dataset(analysis_with_history_file):
    '''
    INPUT
    analysis_with_history_file - the pickle file with expanded KC and success details

    OUTPUT
    None

    This function builds a correlation between the performance on KCs and other factors like incorrects, hints etc.
    The idea is to trace the learning progression students
    '''
    df = pd.read_pickle(analysis_with_history_file)

    row_index = 0
    df_kc_vals = df['KC(Default)'].str.split("~~")
    df_kc_vals = df_kc_vals.reset_index(drop=True)

    print('SUC_Edit Algebraic k' in df.iloc[277692].index)  # SUC_Edit Algebraic k

    KCs = []
    skill_level = []
    incorrects = []
    hints = []
    curr_links = []
    correct_first = []
    student_ids = []
    for row in df_kc_vals:
        print(row_index)
        if not str(row) == 'nan':
            inter_index = 0
            for row_val in row:
                repl_value = row_val.replace('[SkillRule: ', '')
                col_vals = repl_value.split(";")
                col = col_vals[0]

                if 'SUC_' + col in df.iloc[row_index].index:
                    skill_level.append(df.iloc[row_index]['SUC_' + col])
                    incorrects.append(df.iloc[row_index]['Incorrects'])
                    hints.append(df.iloc[row_index]['Hints'])
                    curr_links.append(df.iloc[row_index]['Problem Hierarchy'])
                    correct_first.append(df.iloc[row_index]['Correct First Attempt'])
                    student_ids.append(df.iloc[row_index]['Anon Student Id'])
                    KCs.append(col)
                inter_index += 1
        row_index += 1

    data = {'kc': KCs,
            'skill': skill_level,
            'incorrect': incorrects,
            'hint': hints,
            'curriculum link': curr_links,
            'correct first attempt': correct_first,
            'student id': student_ids}

    df_new = pd.DataFrame(data)
    df_new.to_pickle('output/correlation_df.pickle')
    print('++++++++++++ Finished +++++++++++++++++')


def test_expand_KC_with_history(analysis_with_history_file):

    #Ensure that accumulated success rates are calculated correctly
    df = pd.read_pickle(analysis_with_history_file)

    print(df.iloc[1000]['KC(Default)'])
    df_temp = df.iloc[1000][['Anon Student Id', 'SUC_Remove positive coefficient', 'SUC_Remove coefficient']]
    val1 = df_temp['SUC_Remove positive coefficient']
    val2 = df_temp['SUC_Remove coefficient']
    df_prev = df.iloc[0:1000]
    current_succ_rate = df_prev[(df_prev['Remove positive coefficient'] == 1)].groupby(['Remove positive coefficient'])[
        'Correct First Attempt'].mean()
    test1 = current_succ_rate.values[0]

    current_succ_rate = df_prev[(df_prev['Remove coefficient'] == 1)].groupby(['Remove coefficient'])[
        'Correct First Attempt'].mean()
    test2 = current_succ_rate.values[0]

    assert val1 == test1
    assert val2 == test2



def main():
    training_data_file = 'input/algebra_2005_2006_train.txt'
    analysis_with_history_file = 'output/analysis_with_history_df.pickle'
    overview_data(training_data_file)
    expand_KC_with_history(training_data_file)
    create_correlation_dataset(analysis_with_history_file)
    test_expand_KC_with_history(analysis_with_history_file)

if __name__ == '__main__':
    main()