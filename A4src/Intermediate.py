import sys
import re
# Ticket routing agent system

print("Hi there! Please answer the following questions so that your ticket request can be routed to the correct team: \n\n")

# Create an empty array
answers = []
early_prediction_prompt = "Would you like us to assign you to a team now? (Yes/No/q) \nPS: The more questions you " \
                          "answer," + " the more likely we will assign you to a correct team\n"


def early_prediction():
    while True:
        try:
            answer_2 = input(early_prediction_prompt)
        except ValueError:
            print("Sorry, that is not a valid response")
            continue

        if answer_2 == 'q':
            sys.exit()
        elif re.search(r'\bno\b', answer_2, re.I):
            break
        elif re.search(r'\byes\b', answer_2, re.I):
            print("Your request will be assigned to THIS team:")
            sys.exit()
        elif not (re.search(r'\byes\b', answer_2, re.I) or re.search(r'\bno\b', answer_2, re.I)):
            print("Sorry, you can only give a yes or no response")
            continue
        else:
            break


def get_user_input(prompt):
    while True:
        try:
            answer = input(prompt)
        except ValueError:
            print("Sorry, that is not a valid response")
            continue

        if answer == 'q':
            sys.exit()
        elif not (re.search(r'\byes\b', answer, re.I) or re.search(r'\bno\b', answer, re.I)):
            print("Sorry, you can only give a yes or no response")
            continue
        else:
            answers.append(answer)
            break

    early_prediction()

    return answers


# Questions
request = get_user_input("Request? (Yes/No/q) \n")
incident = get_user_input("Incident? (Yes/No/q) \n")
web = get_user_input("WebServices? (Yes/No/q) \n")
login = get_user_input("Login? (Yes/No/q) \n")
wireless = get_user_input("Wireless (Yes/No/q) \n")
printing = get_user_input("Printing? (Yes/No/q) \n")
idCards = get_user_input("ID Cards? (Yes/No/q) \n")
staff = get_user_input("Staff? (Yes/No/q) \n")
students = get_user_input("Students? (Yes/No/q) \n")






