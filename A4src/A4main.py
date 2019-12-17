from A4src.Basic import *
import sys

# Setting up command line arguments
args = sys.argv
agent = args[1]

print(args)


# function for choosing the correct agent
def choose_agent(argument):
    switcher = {
        'Bas': basic,
        'Int': "Intermediate",
        'Adv': "Advanced",
    }
# Get the function from switcher dictionary
    func = switcher.get(argument, lambda: "Invalid option")
    # Execute the function
    print(func())


# Choose the agent
choose_agent(agent)
