import sys
import re
from Basic import *
from numpy import array


"""
Python file for all the methods and variables used by the Ticketing Routing Agent
"""""

answers = []  # Create an empty array for the answers

# load the model
clf = joblib.load('../hidden_unit_models_final/mynetwork_5.joblib') # load the model





def get_user_input(prompt,  p=True):
    """
    Gets the user's input. 
    :return: answer
    """""
    while True:
        try:
            answer = input(prompt).lower().strip()
        except ValueError:
            print("Sorry, that is not a valid response")
            continue

        if answer.lower() == 'q':
            sys.exit()
        elif (p == True) & (answer == 'p'):
            break
        elif not (re.search(r'\byes\b', answer, re.I) or re.search(r'\bno\b', answer, re.I)):
            print("Sorry, that is not a valid response")
            continue
        else:
            answers.append(answer.lower().strip())
            break

    return answer


def convert_answers(answers):
    """
    Converts the answers from the model into one hot encoding using numpy boolean indexing
    :return: X_2
    """""
    X_2 = np.reshape(array(answers), (-1, 9))
    no_bool = X_2 == 'no'
    yes_bool = X_2 == 'yes'
    X_2[no_bool] = 0
    X_2[yes_bool] = 1
    X_2 = X_2.astype(int)
    return X_2


def pick_team(team):
    """
    Picks a response team
    :return: response_team
    """""
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


def pick_team_encoded(team):
    """
    Picks a response team. Returns in one-hot code encoded form
    :return: response_team
    """""
    teams = {
        0: [1, 0, 0, 0, 0],
        1: [0, 1, 0, 0, 0],
        2: [0, 0, 1, 0, 0],
        3: [0, 0, 0, 1, 0],
        4: [0, 0, 0, 0, 1],
    }
    #get teams from dictionary
    response_team = teams.get(team, lambda: "Invalid month")
    return response_team


def give_options(team):
    """
    Gives the user response team options based on what was wrongly predicted before
    :return: option
    """""
    switcher = {
        0: '1 - Datawarehouse\n2 - Emergencies\n3 - Equipment\n4 - Networking\n',
        1: '0 - Credentials\n2 - Emergencies\n3 - Equipment\n4 - Networking\n',
        2: '0 - Credentials\n1 - Datawarehouse\n3 - Equipment\n4 - Networking\n',
        3: '0 - Credentials\n1 - Datawarehouse\n2 - Emergencies\n4 - Networking\n',
        4: '0 - Credentials\n1 - Datawarehouse\n2 - Emergencies\n3 - Equipment\n',
    }
    option = switcher.get(team, lambda: "Invalid option")
    return option


def prediction(clf, X_2):
    """
    Makes a prediction based on the answers provided
    :return: [pick_team(team), team]
    """""
    no_allocation = np.array([[0, 0, 0, 0, 0]])  # case where the probability does not exceed 0.5
    prediction_int = clf.predict(X_2)
    prediction_probab = clf.predict_proba(X_2)
    if np.array_equal(prediction_int, no_allocation):
        team = np.argmax(prediction_probab)  # takes the maximum probability if all probabilities are < 0.5
    else:
        team = np.argmax(prediction_int)

    return [pick_team(team), team]


def fill_missing_columns(answers):
    """
    During an early prediction, fills in the missing columns by using the mode
    :return: answers
    """""
    input_cols = array(input_values.columns)
    answered_dict = {}  # dictionary of answers provided so far
    no_answers = answers.__len__()
    counter = 0 # counter for each iteration of the for loop
    # for loop for creating the dictionary
    for col, answer in zip(input_cols, answers):
        answered_dict[col] = answer.capitalize()
        counter += 1
        if counter == no_answers:
            break

    # Filters the inputs CV columns based on the answers provided so far
    filtered = input_values[np.logical_and.reduce([(input_values[k] == v) for k, v in answered_dict.items()])]
    # Finds the most frequently occurring values of the missing columns based on the answers provided so far

    mode_columns = array(filtered.mode().iloc[:, no_answers:9])
    if(mode_columns.size > 11):
        mode_columns = np.delete(mode_columns,(1),axis=0)
    # Adds the most frequently occurring answers back into the answers array for prediction
    for col in mode_columns:
        for value in col:
            answers.append(str(value).lower())

    return answers


def new_ticket_request():
    """
    Asks for another ticket request
    :return: 
    """""
    while True:
        try:
            another_ticket = input("Would you like another ticket request? (Yes/No)\n")
        except ValueError:
            print("Sorry, that is not a valid response")
            continue
        if another_ticket.lower() == 'no':
            sys.exit()
        elif another_ticket.lower() == 'yes':
            del answers[:]  # empties the answer list for a new round of questions
            intermediate()  # Re-run's the new ticket process
            break
        elif not (re.search(r'\byes\b', another_ticket, re.I) or re.search(r'\bno\b', another_ticket, re.I)):
            print("Sorry, that is not a valid response\n")
            continue


def team_allocate(predicted_team_number):
    """
    Allocates a team to the user based on the option chosen
    :return: selected_team_encoded
    """""
    while True:
        try:
            team_select = input(
                "Apologies for that, which team from below would you like to speak to? Please select a number:\n"
                + give_options(predicted_team_number))
        except ValueError:
            print("Sorry, that is not a valid response")
            continue
        if int(team_select) == predicted_team_number:
            print("Sorry, we already allocated this team, please pick one available from the list:")
            continue
        elif int(team_select) not in [0,1,2,3,4]:
            print("Sorry, that is not a valid response")
            continue
        else:
            break
    # processes the selected team
    selected_team = pick_team(int(team_select))
    selected_team_encoded = pick_team_encoded(int(team_select))
    print("Thank you for your patience, your request will be sent to the " + selected_team + " team.\n")
    return selected_team_encoded


def check_if_happy(new_ticket, predicted_team_number):
    """
    Checks if the user is happy with the prediction
    :return: 
    """""
    while True:
        try:
            # check if user is happy
            happy_q = input("Are you happy with this allocation?(Yes/No)\n").lower()
        except ValueError:
            print("Sorry, that is not a valid response")
            continue
        if happy_q == 'no':
            return team_allocate(predicted_team_number)
            break
        elif happy_q == 'yes':
            print("Have a nice day!\n")
            return True
            break

        elif not (re.search(r'\byes\b', happy_q, re.I) or re.search(r'\bno\b', happy_q, re.I)):
            print("Sorry, that is not a valid response\n")
            continue


def retrain(new_ticket, selected_team_encoded):
    """
    Retrains the model based on the new full list of answers and the selected team
    :return: 
    """""
    # Adding new ticket to the training tables
    updated_X_train = np.concatenate((X_train, new_ticket), axis=0)
    selected_team_encoded = np.array(selected_team_encoded)
    updated_y_train = np.vstack([y_train, selected_team_encoded])
    # Retrain the model
    clf.set_params(verbose=True, tol=0.0001, n_iter_no_change=1000, max_iter=20000)
    clf.fit(updated_X_train, updated_y_train)
    print("Our system, has learnt from your input.  \n" +
          " (Model has been retrained...)\n")


def get_feedback(answer_count, answers_so_far):
    """
    Runs a series of feedback questions to get answers for improving the model after an early prediction
    :return: new_ticket
    """""
    del answers[:]  # empty answers
    print("To help improve our system please answer the remaining question(s)\n")
    for feedback in feedback_args[answer_count:]:  # get feedback based on the remaining questions
        get_user_input(*feedback)
    new_ticket = np.concatenate((answers, answers_so_far[0:answer_count]), axis=0)
    new_ticket = convert_answers(new_ticket)
    return new_ticket


def intermediate():
    """
    Runs the ticketing routing agent
    :return: 
    """""
    print(opening_message)
    for question in question_args:
        if get_user_input(*question) == 'p':  # code block for an early prediction
            answer_count = len(answers)
            answers_so_far = answers
            predicted_columns = fill_missing_columns(answers)  # combines the answers given with the mode of missing columns
            converted_p = convert_answers(predicted_columns)  # one-hot encodes the answer set
            predicted_team = prediction(clf, converted_p)[0]  # makes a prediction based on answer set, return team
            predicted_team_number = prediction(clf, converted_p)[1]  # returns the numerical rep. of team
            print("Based on your answers, your request will be sent to the " + predicted_team + " team.")
            happy = check_if_happy(converted_p, predicted_team_number)
            if happy == True:
                break  # if happy, break
            else:
                feedback_answers = get_feedback(answer_count, answers_so_far) # if not happy, get feedback and retrain
                retrain(feedback_answers, happy)
            break

        elif len(answers) == 9: # code block for when the user answers all 9 questions
            converted = convert_answers(answers)  # ""
            predicted_team = prediction(clf, converted)[0]  # ""
            predicted_team_number = prediction(clf, converted)[1]  # ""
            print("Based on your answers, your request will be sent to the " + predicted_team + " team.")
            happy = check_if_happy(converted, predicted_team_number)
            if happy == True:
                break
            else:
                retrain(converted, happy)
            break

    new_ticket_request()

















