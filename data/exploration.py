import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import plotly.express as px


def plot_success_by_KC_distribution(analysis_with_history_file):
    '''
   INPUT
   analysis_with_history_file - the pickle file with expanded KC and success details

   OUTPUT
   None

   This functions builds a stackbar chart that would display the success/failure counts per each knowledge component.
   Idea is to get a sense of how KC performances are distributed
   '''

    df = pd.read_pickle(analysis_with_history_file)

    plt_data = []
    cols = [c for c in df.columns if c.lower()[:4] != 'suc_']
    for col in cols[19:]:
        total_occ = df[df[col] == 1].shape[0]
        succ_occ = df[(df[col] == 1) & (df['Correct First Attempt'] == 1)].shape[0]

        plt_data.append([col, 'Correct', succ_occ])
        plt_data.append([col, 'Incorrect', (total_occ - succ_occ)])

    draw_stack_bar_plot(plt_data, "KC", "First Attempt", "Type")


def plot_hints_incorrects_by_KC_distribution(analysis_with_history_file):
    '''
   INPUT
   analysis_with_history_file - the pickle file with expanded KC and success details

   OUTPUT
   None

   This functions builds a stackbar chart that would display the hints/incorrects per knowledge component
   Idea is to get a sense of how students have struggled in specific KCs
   '''
    df = pd.read_pickle(analysis_with_history_file)
    plt_data = []

    cols = [c for c in df.columns if c.lower()[:4] != 'suc_']
    for col in cols[19:]:
        incorrects = df[df[col] == 1].groupby(col)['Incorrects'].mean().iloc(0)[0]
        hints = df[df[col] == 1].groupby(col)['Hints'].mean().iloc(0)[0]


        plt_data.append([col, 'Incorrects', round(incorrects, 2)])
        plt_data.append([col, 'Hints', round(hints, 2)])

    draw_stack_bar_plot(plt_data, "KC", "Measurement", "Type")


def plot_issues_by_curriculum_link(analysis_with_history_file):
    '''
   INPUT
   analysis_with_history_file - the pickle file with expanded KC and success details

   OUTPUT
   None

   This functions builds a stackbar chart that would plot the mean number of incorrect and hints per problem hierarchy
   Idea is to find the curriculum sections the students have struggled with
   '''
    df = pd.read_pickle(analysis_with_history_file)
    df_mean = df.groupby(['Problem Hierarchy'])['Incorrects', 'Hints'].mean()
    plt_data = []
    for index, row in df_mean.iterrows():
        plt_data.append([index, 'Incorrects', row['Incorrects']])
        plt_data.append([index, 'Hints', row['Hints']])
    draw_stack_bar_plot(plt_data, "Problem Hierarchy", "Measurement", "Type")


def plot_correlation_skill_success(correlation_file, student_no):
    '''
   INPUT
   correlation_file - file with the performance correlation data
   student_no - student identifier

   OUTPUT
   None

   Function builds a scatter plot that shows the the learning progression of students
   '''
    df = pd.read_pickle(correlation_file)
    df_plot = df[df['student id'] == student_no]
    df_plot['sequence'] = df_plot.index.values
    draw_scatter_plot_simple(df_plot, 'sequence', 'skill', 'kc')



def students_with_best_outcomes(analysis_with_history_file, num_students, ascending):
    '''
   INPUT
   num_students - number of records needed
   ascending - sort order. Depends on whether you are looking for best students or worst students

   OUTPUT
   None

   Print the students with the best/worst first attempt success rates
   '''
    df = pd.read_pickle(analysis_with_history_file)
    df_stu = df.groupby(['Anon Student Id'])['Correct First Attempt'].mean().sort_values(ascending=ascending)
    print(df_stu.head(num_students))


def plot_time_spent_correlation(analysis_with_history_file):
    '''
   INPUT
   analysis_with_history_file - the pickle file with expanded KC and success details

   OUTPUT
   None

   Plot the correlation between the average time spent on a problem and the average first time success rates
   The idea is to investigate if more time is spent on a problem whether it is an indication of a difficult problem
   '''
    df = pd.read_pickle(analysis_with_history_file)
    df_view = df.groupby('Problem Name').agg({'Correct First Attempt': ['mean'], 'Step Duration (sec)': ['mean']})

    data = {'Problem Name': df_view.index,
            'Mean Correct First Attempt': df_view['Correct First Attempt']['mean'].values,
            'Mean Step Duration (sec)': df_view['Step Duration (sec)']['mean'].values}

    df_view = pd.DataFrame(data)

    df_view = df_view[df_view['Mean Correct First Attempt'] >= 0.4]
    draw_scatter_plot_simple(df_view, 'Mean Step Duration (sec)', 'Mean Correct First Attempt', 'Problem Name')

#Draws a stackbar chart. Note the bin size to decide on the number of records per page
def draw_stack_bar_plot(plt_data, x, y, color):
    bin_size = 120
    bins, remainder = divmod(len(plt_data), bin_size)
    count = 0
    for bin in range(bins):
        df_plot = pd.DataFrame(columns=[x, color, y],
                               data=plt_data[count * bin_size:(count * bin_size) + bin_size - 1])
        fig = px.bar(df_plot, x=x, y=y, color=color)
        fig.show()
        count += 1

    df_plot = pd.DataFrame(columns=[x, color, y],
                           data=plt_data[count * bin_size:(count * bin_size) + remainder])
    fig = px.bar(df_plot, x=x, y=y, color=color)
    fig.show()

#draws a scatter plot with color, dot size and also hover information
def draw_scatter_plot(df, x, y, color, size, hover):
    fig = px.scatter(df, x=x, y=y, color=color,
                     size=size, hover_data=[hover])
    fig.show()

#draws a simple scatter plot
def draw_scatter_plot_simple(df, x, y, color):
    fig = px.scatter(df, x=x, y=y, color=color)
    fig.show()


def main():
    corelation_data_file = 'output/correlation_df.pickle'
    analysis_with_history_file = 'output/analysis_with_history_df.pickle'

    plot_success_by_KC_distribution(analysis_with_history_file)
    plot_hints_incorrects_by_KC_distribution(analysis_with_history_file)
    plot_issues_by_curriculum_link(analysis_with_history_file)
    plot_time_spent_correlation(analysis_with_history_file)
    students_with_best_outcomes(analysis_with_history_file, 100, True)
    students_with_best_outcomes(analysis_with_history_file, 100, False)
    plot_correlation_skill_success(corelation_data_file,'Ds3B2dRQo8')
    plot_correlation_skill_success(corelation_data_file,'LM96hW22T2')


if __name__ == '__main__':
    main()




