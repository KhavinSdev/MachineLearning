import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

df = pd.read_csv("diabetes[1].csv")

df_positives = df.loc[df['Outcome'] == 1]

feature_vectors = df_positives[['Pregnancies', 'Age']].to_numpy()
output_vectors = df_positives[['BMI']].to_numpy()
w_vector = np.zeros(len(feature_vectors[0]), )

print(output_vectors)

def w_derivative(w_vector, x_vectors, b):

    sum = 0

    for i in range(len(x_vectors)):

        f_value = np.dot(w_vector, x_vectors[i]) + b
        sum = sum + (f_value - output_vectors[i]) * x_vectors[i]

    return sum/len(x_vectors)

def b_derivative(w_vector, x_vectors, b):

    sum = 0

    for i in range(len(x_vectors)):

        f_value = np.dot(w_vector, x_vectors[i]) + b
        sum = sum + (f_value - output_vectors[i]) 
    
    return sum/len(x_vectors)
