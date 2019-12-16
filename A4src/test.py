from A4src.Basic import *

cols = {"Request": 'No'}
questions_answered = 1
filtered = inputs[np.logical_and.reduce([(inputs[k] == v) for k,v in cols.items()])]
mode_columns = array(filtered.mode().iloc[:, questions_answered:9])

print(mode_columns)

