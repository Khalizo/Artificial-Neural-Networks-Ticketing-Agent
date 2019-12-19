from Basic import *
from Intermediate import *
from Advanced import *
import sys

# Setting up command line arguments
args = sys.argv
agent = args[1]


def choose_agent(argument):
    """
    Runs the agent of choice
    :return: 
    """""
    switcher = {
        'Bas': basic,
        'Int': intermediate,
        'Adv': advanced,
    }
    func = switcher.get(argument, lambda: "Invalid option")
    return func()


# Choose the agent
choose_agent(agent)
