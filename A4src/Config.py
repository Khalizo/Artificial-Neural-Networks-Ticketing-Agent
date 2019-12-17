

# opening message

opening_message = "Hi there! Please answer the following 9 questions so that your ticket request can be routed" \
                  " to the correct team.\n"+ "If you'd like us to make a prediction, please answer at least 3 " \
                 "questions\n" + "Please answer the questions with " \
                  "either Yes, No, P or Q.\n" + "P = early prediction. Please note, that we will need at" \
                  " least 3 answers before " \
                  "this is available\n" + "Q = quit\n\n"


# Params for models with varying hidden unit sizes
params = [{'hidden_layer_sizes': (1,)}, {'hidden_layer_sizes': (2,)}, {'hidden_layer_sizes': (3,)},
          {'hidden_layer_sizes': (4,)}, {'hidden_layer_sizes': (5,)}, {'hidden_layer_sizes': (6,)},
          {'hidden_layer_sizes': (7,)}, {'hidden_layer_sizes': (8,)}, {'hidden_layer_sizes': (9,)},
          {'hidden_layer_sizes': (10,)}]

# Files for saving the networks and results
saved_networks = ['../hidden_unit_models/mynetwork_1.joblib', '../hidden_unit_models/mynetwork_2.joblib',
                  '../hidden_unit_models/mynetwork_3.joblib', '../hidden_unit_models/mynetwork_4.joblib',
                  '../hidden_unit_models/mynetwork_5.joblib', '../hidden_unit_models/mynetwork_6.joblib',
                  '../hidden_unit_models/mynetwork_7.joblib', '../hidden_unit_models/mynetwork_8.joblib',
                  '../hidden_unit_models/mynetwork_9.joblib', '../hidden_unit_models/mynetwork_10.joblib']
save_fig = '../demo_results/hidden_unit_model_fig.png'
model_results = '../demo_results/hidden_unit_model_results.csv'

# Labels for the hidden unit graph plot
labels = ["1 hidden unit", "2 hidden units", "3 hidden units", "4 hidden units", "5 hidden units",
          "6 hidden units", "7 hidden units", "8 hidden units", "9 hidden units", "10 hidden units"]

# Arguments for the hidden unit graph plot
plot_args = [{'c': 'red', 'linestyle': '-'},
             {'c': 'red', 'linestyle': '--'},
             {'c': 'blue', 'linestyle': '-'},
             {'c': 'blue', 'linestyle': '--'},
             {'c': 'green', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '--'},
             {'c': 'black', 'linestyle': '-'},
             {'c': 'black', 'linestyle': '--'},
             {'c': 'magenta', 'linestyle': '-'},
             {'c': 'magenta', 'linestyle': '--'}]

# Arguments for the questions asked to get the user's input
question_args = [["Question 1: Request? (Yes/No/Q) \n", False],
                 ["Question 2: Incident? (Yes/No/Q) \n", False],
                 ["Question 3: WebServices? (Yes/No/Q) \n", False],
                 ["Question 4: Login? (Yes/No/P/Q) \n", True],
                 ["Question 5: Wireless (Yes/No/P/Q) \n", True],
                 ["Question 6: Printing? (Yes/No/P/Q) \n", True],
                 ["Question 7: ID Cards? (Yes/No/P/Q) \n", True],
                 ["Question 8: Staff? (Yes/No/P/Q) \n", True],
                 ["Question 9: Students? (Yes/No/P/Q) \n", True]]

# List of all the columns of the inputs
feedback_args = [["Question 1: Request? (Yes/No/Q) \n", False],
                 ["Question 2: Incident? (Yes/No/Q) \n", False],
                 ["Question 3: WebServices? (Yes/No/Q) \n", False],
                 ["Question 4 (Feedback): Login? (Yes/No/Q) \n", False],
                 ["Question 5 (Feedback): Wireless (Yes/No/Q) \n", False],
                 ["Question 6 (Feedback): Printing? (Yes/No/Q) \n", False],
                 ["Question 7 (Feedback): ID Cards? (Yes/No/Q) \n", False],
                 ["Question 8 (Feedback): Staff? (Yes/No/P/Q) \n", False],
                 ["Question 9 (Feedback): Students? (Yes/No/Q) \n", False]]


