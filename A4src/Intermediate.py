import sys
import re
from A4src.Basic import *
from numpy import array
from sklearn.neural_network import MLPClassifier
# Ticket routing agent system



# Create an empty array for the answers
answers = []
# message before an early prediction
early_prediction_prompt = "Would you like us to assign you to a team now? (Yes/No/q) \nPS: The more questions you " \
                          "answer," + " the more likely we will assign you to a correct team\n"
# load the model
clf = joblib.load('../hidden_unit_models_final/mynetwork_5.joblib')


def get_user_input(prompt,  p=True):

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


def pick_team_encoded(team):
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
        switcher = {
            0: '1 - Datawarehouse\n2 - Emergencies\n3 - Equipment\n4 - Networking\n',
            1: '0 - Credentials\n2 - Emergencies\n3 - Equipment\n4 - Networking\n',
            2: '0 - Credentials\n1 - Datawarehouse\n3 - Equipment\n4 - Networking\n',
            3: '0 - Credentials\n1 - Datawarehouse\n2 - Emergencies\n4 - Networking\n',
            4: '0 - Credentials\n1 - Datawarehouse\n2 - Emergencies\n3 - Equipment\n',
        }
        # Get the function from switcher dictionary
        func = switcher.get(team, lambda: "Invalid option")
        # Execute the function
        return func


def prediction(clf, X_2):
    no_allocation = np.array([[0, 0, 0, 0, 0]])
    prediction_int = clf.predict(X_2)
    prediction_probab = clf.predict_proba(X_2)
    if np.array_equal(prediction_int, no_allocation):
        team = np.argmax(prediction_probab)
    else:
        team = np.argmax(prediction_int)

    return [pick_team(team), team]


def fill_missing_columns(answers):
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
    # Adds the most frequently occurring answers back into the answers array for prediction
    for col in mode_columns:
        for value in col:
            answers.append(value.lower())

    return answers


def new_ticket_request():
    while True:
        try:
            another_ticket = input("Would you like another ticket request? (Yes/No)\n")
        except ValueError:
            print("Sorry, that is not a valid response")
            continue
        if another_ticket.lower() == 'no':
            sys.exit()
        elif another_ticket.lower() == 'yes':
            del answers[:]
            intermediate()
            break
        elif not (re.search(r'\byes\b', another_ticket, re.I) or re.search(r'\bno\b', another_ticket, re.I)):
            print("Sorry, that is not a valid response\n")
            continue


def team_allocate(predicted_team_number):
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
    selected_team = pick_team(int(team_select))
    selected_team_encoded = pick_team_encoded(int(team_select))
    print("Thank you for your patience, your request will be sent to the " + selected_team + " team.\n"
        "To help improve our system please answer the remaining question(s)")
    return selected_team_encoded


def check_if_happy(new_ticket, predicted_team_number):

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
    # Adding new ticket to the training tables
    updated_X_train = np.concatenate((X_train, new_ticket), axis=0)
    updated_y_train = np.vstack([y_train, selected_team_encoded])
    # Retrain the model
    clf.fit(updated_X_train, updated_y_train)
    print("Thank you very much for your time! Our system, has learnt from your input.  \n" +
          " (Model has been retrained...)\n")


def get_feedback(answer_count, answers_so_far):
    del answers[:]  # empty answers
    for feedback in feedback_args[answer_count:]:  # get feedback based on the remaining questions
        get_user_input(*feedback)
    new_ticket = np.concatenate((answers, answers_so_far[0:answer_count]), axis=0)
    new_ticket = convert_answers(new_ticket)
    return new_ticket


def intermediate():
    i = 0
    print(opening_message)
    while i < len(question_args):
        if get_user_input(*question_args[i]) == 'p':
            answer_count = len(answers)
            answers_so_far = answers
            predicted_columns = fill_missing_columns(answers)
            converted_p = convert_answers(predicted_columns)
            predicted_team = prediction(clf, converted_p)[0]
            predicted_team_number = prediction(clf, converted_p)[1]
            print("Based on your answers, your request will be sent to the " + predicted_team + " team.")
            happy = check_if_happy(converted_p, predicted_team_number)
            if happy == True:
                break
            else:
                feedback_answers = get_feedback(answer_count, answers_so_far)
                retrain(feedback_answers, happy)
            break

        elif len(answers) == 9:
            converted = convert_answers(answers)
            prediction(clf, converted)
        i += 1

    new_ticket_request()

















