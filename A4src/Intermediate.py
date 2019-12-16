import sys
import re
from sklearn.externals import joblib
import numpy as np
from A4src.config import *
from numpy import array
from sklearn.neural_network import MLPClassifier
# Ticket routing agent system

print("Hi there! Please answer the following 9 questions so that your ticket request can be routed to the correct team.\n"
      "If you'd like us to make a prediction, please answer at least 3 questions\n"
      "Please answer the questions with either Yes, No, P or Q.\n"
      "P = early prediction. Please note, that we will need at least 3 answers before this is available\n"
      "Q = quit\n\n")

# Create an empty array for the answers
answers = []
# message before an early prediction
early_prediction_prompt = "Would you like us to assign you to a team now? (Yes/No/q) \nPS: The more questions you " \
                          "answer," + " the more likely we will assign you to a correct team\n"
# load the model
clf = joblib.load('../hidden_unit_models_final/mynetwork_5.joblib')


def get_user_input(prompt, p=True):
    while True:
        try:
            answer = input(prompt)
        except ValueError:
            print("Sorry, that is not a valid response")
            continue

        if answer.lower() == 'q':
            sys.exit()
        elif (p == True) & (answer.lower() == 'p'):
            print('prediction')
            break
        elif not (re.search(r'\byes\b', answer, re.I) or re.search(r'\bno\b', answer, re.I)):
            print("Sorry, that is not a valid response")
            continue
        else:
            answers.append(answer.lower().strip())
            break

    return answer


# convert the answers from the model into one hot encoding using numpy boolean indexing
def convert_answers(answers):
    X_2 = np.reshape(array(answers), (-1, 9))
    no_bool = X_2 == 'no'
    yes_bool = X_2 == 'yes'
    X_2[no_bool] = 0
    X_2[yes_bool] = 1
    X_2 = X_2.astype(int)
    return X_2


def pick_team(team):
    teams = {
        0: 'Credentials',
        1: 'Datawarehouse',
        2: 'Emergencies',
        3: 'Equipment',
        4: 'Networking',
    }
    #get teams from dictionary
    response_team = teams.get(team, lambda: "Invalid month")
    return response_team


def prediction(clf,X_2):
    no_allocation = np.array([[0, 0, 0, 0, 0]])
    prediction_int = clf.predict(X_2)
    prediction_probab = clf.predict_proba(X_2)
    if np.array_equal(prediction_int, no_allocation):
        team = np.argmax(prediction_probab)
    else:
        team = np.argmax(prediction_int)

    return print("Based on your answers, your request will be sent to the " + pick_team(team) + " team.")


def ask_questions():
    for question in question_args:
        if get_user_input(*question) == 'p':
            break
            print(answers.__len__())

    converted = convert_answers(answers)
    prediction(clf, converted)


ask_questions()











