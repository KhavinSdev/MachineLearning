import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

df = pd.read_csv("diabetes[1].csv")

df_positives = df.loc[df['Outcome'] == 1]


array_y = df_positives.loc[:,'Pregnancies'].to_numpy()
array_age = df_positives.loc[:, 'Age'].to_numpy()

# plt.scatter(array_age, array_y)
# plt.show()

def partial_w(w, b, array_x, array_y):
    sum = 0
    for i in range(len(array_x)):
        sum = sum + ((w * array_x[i] + b) - array_y[i]) * array_x[i] 

    return sum/len(array_x)

def partial_b(w, b, array_x, array_y):
    sum = 0
    for i in range(len(array_x)):
        sum = sum + ((w * array_x[i] + b) - array_y[i]) 

    return sum/len(array_x)

def gradient_descent_implementation(w, b, learning_rate, array_x, array_y):

    while True:
        partial_wrt_w = partial_w(w, b, array_x, array_y)
        partial_wrt_b = partial_b(w, b, array_x, array_y)

        w = w - learning_rate * partial_wrt_w
        b = b - learning_rate * partial_wrt_b

        print(f"y = {str(w)}x + {str(b)}")

        # time.sleep(2)

gradient_descent_implementation(0, 0, 0.001, array_age, array_y)

