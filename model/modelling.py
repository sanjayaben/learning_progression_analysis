import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def model_logistic_regression():
    '''
    INPUT
    None

    OUTPUT
    None

    loads the prepared dataset from the feature_engineering steps and performs a Logistic Regression for binary classification

    '''
    X_train, X_test, y_train, y_test = load_data(0.3, 42)

    model = LogisticRegression(random_state=0)  # Instantiate
    model.fit(X_train, y_train)  # Fit

    # Predict and score the model
    y_test_preds = model.predict(X_test)
    print("The accuracy score for the model was {} on {} values.".format(accuracy_score(y_test, y_test_preds), len(y_test)))
    print("Confusion Matrix = {} ".format(confusion_matrix(y_test, y_test_preds)))
    plot_confusion_matrix(y_test, y_test_preds)

def plot_confusion_matrix(y_true, y_pred):
    data = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(data, columns=np.unique(y_true), index=np.unique(y_true))
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g', annot_kws={"size": 16})  # font size
    plt.show()

def model_svm():
    '''
    INPUT
    None

    OUTPUT
    None

    loads the prepared dataset from the feature_engineering steps and performs a SVM for binary classification

    '''

    X_train, X_test, y_train, y_test = load_data(0.3, 42)

    model = SVC(gamma='auto') # Instantiate
    model.fit(X_train, y_train)  # Fit

    # Predict and score the model
    y_test_preds = model.predict(X_test)
    print("The accuracy score for the model was {} on {} values.".format(accuracy_score(y_test, y_test_preds), len(y_test)))

def model_knn():
    '''
    INPUT
    None

    OUTPUT
    None

    loads the prepared dataset from the feature_engineering steps and performs a knn for binary classification

    '''
    X_train, X_test, y_train, y_test = load_data(0.3, 42)

    model =  KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)  # Fit

    # Predict and score the model
    y_test_preds = model.predict(X_test)
    print("The accuracy score for the model was {} on {} values.".format(accuracy_score(y_test, y_test_preds), len(y_test)))

def load_data(test_split, rnd_state):
    '''
    INPUT
    test_split - train/test split ratio
    rnd_state - random state indicator

    OUTPUT
    None

    loads the prepared dataset from the feature_engineering steps and performs a Logistic Regression for binary classification

    '''
    df = pd.read_pickle('../data/output/training_data_final.pickle')
    y = df['Correct First Attempt'].apply(str)
    X = df.drop(['Correct First Attempt'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=rnd_state)
    return X_train, X_test, y_train, y_test




def tune_hyper_parameters():
    '''
    INPUT
    None

    OUTPUT
    None

    This function does the following
    1. Creates a pipeline that includes a LogisticRegression
    2. Define a set of parameter value ranges for estimator
    3. Create and return a GridSearchCV
    '''
    pipeline = Pipeline([
        ('clf', LogisticRegression(max_iter = 4000))])

    parameters = {
    'clf__penalty' : ['l2'],
    'clf__C' : [0.1, 1, 10],
    'clf__solver' : ['lbfgs','saga']}

    X_train, X_test, y_train, y_test = load_data(0.3, 42)
    cv = GridSearchCV(pipeline, param_grid=parameters)
    cv.fit(X_train, y_train)
    y_test_preds = cv.predict(X_test)
    print("The accuracy score for the model was {} on {} values.".format(accuracy_score(y_test, y_test_preds),
                                                                         len(y_test)))
    print(cv.best_params_)



def main():
    model_logistic_regression()
    # model_svm()
    # model_knn()
    # tune_hyper_parameters()


if __name__ == '__main__':
    main()