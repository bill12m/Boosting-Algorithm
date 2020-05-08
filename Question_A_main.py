import pandas as pd
import numpy as np
import subprocess as sp

sp.call('clear', shell = True)

df = pd.DataFrame(np.array([[0.5,-1],[3.0,-1],[4.5,1],[4.6,1],[4.9,1],[5.2,-1],[5.3,-1],
                  [5.5,1],[7.0,-1],[9.5,-1]]), columns = ['x','y'])
df['Distance'] = np.abs(df.iloc[:,0] - 5.0)
df = df.sort_values(by = ['Distance']).reset_index()
del(df['index'])

neighbors = [1,3,5,9]
labels = ["1-nn","3-nn","5-nn","9-nn"]
answers = []

nearness = 0
for neighbor in neighbors:
    sign = 0
    for i in range(neighbor):
        sign += df.loc[i,'y']
    if sign <0:
        answers.append(-1)
    else:
        answers.append(1)
    nearness +=1

answers = pd.DataFrame(answers, index = labels, columns = ['Classification'])
display(answers)