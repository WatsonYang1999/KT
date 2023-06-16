import numpy as np
#import pandas as pd


data = np.load('ednet.npz')

y, skill, problem, real_len = data['y'], data['skill'], data['problem'], data['real_len']

print(problem)
