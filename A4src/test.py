from A4src.Basic import *
from A4src.Intermediate import *


def fill_missing_columns(answers):
    input_cols = array(inputs.columns)
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
    filtered = inputs[np.logical_and.reduce([(inputs[k] == v) for k, v in answered_dict.items()])]
    # Finds the most frequently occurring values of the missing columns based on the answers provided so far
    mode_columns = array(filtered.mode().iloc[:, no_answers:9])
    # Adds the most frequently occurring answers back into the answers array for prediction
    for col in mode_columns:
        for value in col:
            answers.append(value.lower())

    return answers


hello = ['yes', 'no', 'no']

print(fill_missing_columns(hello))

# check if happy

happy_q = input("Are you happy with this allocation?\n").lower()
if happy_q == 'no':
    new_ticket = []
    team_select = input(
        "Apologies for that, which team from below would you like to speak to? Please select a number:\n"
        "0 - Credentials\n1 - Datawarehouse\n2 - Emergencies\n3 - Equipment\n4 - Networking\n")
    selected_team = pick_team(int(team_select))
    for value in hello:
        new_ticket.append(value.capitalize())
    new_ticket.append(selected_team)
    print("Thank for your patience, your request will be sent to the " + selected_team)

print(new_ticket)
