from A4src.Basic import *
from A4src.Intermediate import *


def fill_missing_columns(answers):
    input_cols = array(input_values.columns)
    answered_dict = {}  # dictionary of answers provided so far
    no_answers = answers.__len__()
    counter = 0  # counter for each iteration of the for loop
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


hello = ['yes', 'no', 'no']

print(fill_missing_columns(hello))


def check_if_happy(new_ticket):

    while True:
        try:
            # check if user is happy
            happy_q = input("Are you happy with this allocation?(Yes/No)\n").lower()
        except ValueError:
            print("Sorry, that is not a valid response")
            continue
        if happy_q == 'no':
            team_select = input(
                "Apologies for that, which team from below would you like to speak to? Please select a number:\n"
                "0 - Credentials\n1 - Datawarehouse\n2 - Emergencies\n3 - Equipment\n4 - Networking\n")
            selected_team = pick_team(int(team_select))
            selected_team_encoded = pick_team_encoded(int(team_select))
            print("Thank you for your patience, your request will be sent to the " + selected_team + " team.")
            # Adding new ticket to the training tables
            updated_X_train = np.concatenate((X_train, new_ticket), axis=0)
            updated_y_train = np.vstack([y_train, selected_team_encoded])
            # Retrain the model
            clf.fit(updated_X_train, updated_y_train)
            print("Our system, has learnt from this request.")
            break

        elif happy_q == 'yes':
            print("Have a nice day!")
            break

        elif not (re.search(r'\byes\b', happy_q, re.I) or re.search(r'\bno\b', happy_q, re.I)):
            print("Sorry, that is not a valid response\n")
            continue


