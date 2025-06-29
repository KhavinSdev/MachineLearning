import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

x = np.random.rand(5)
y = np.random.rand(5)

plt.ion()
fig, ax = plt.subplots()
x, y = [],[]
sc = ax.scatter(x,y, s=5)
plt.xlim(0,400)
plt.ylim(0,0.8)
plt.draw()



df = pd.read_csv("housepricez.csv")

# df_positives = df.loc[df['Outcome'] == 1]
df_positives = df

feature_vectors = df_positives[["Bedrooms","Bathrooms","Sqft_living","Sqft_lot","Floors","Grade"]].to_numpy()
output_vectors = df_positives[['Price']].to_numpy()
w_vector = np.zeros(len(feature_vectors[0]), )

feature_means = np.mean(feature_vectors, axis=0)
feature_standardd = np.std(feature_vectors, axis=0)

output_mean = np.mean(output_vectors, axis=0)
output_standardd = np.std(output_vectors, axis=0)

print(output_standardd)


def standard_normalisation(features, output_vectors):
    features = (features - feature_means)/feature_standardd
    
    output_vectors = (output_vectors - output_mean)/output_standardd

    return (features, output_vectors)

feature_vectors, output_vectors = standard_normalisation(feature_vectors, output_vectors)

# print(feature_vectors)
# print(output_vectors)

def cost_calculate(w_vector, x_vectors, b):
    sum = 0

    for i in range(len(x_vectors)):

        f_value = np.dot(w_vector, x_vectors[i]) + b
        sum = sum + (f_value - output_vectors[i]) ** 2

    return sum/len(x_vectors)



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

def gradient_descent(w, b, learning_rate, feature_vectors, margin_of_error):
    trainings = 0
    cost_of_trainin = cost_calculate(w, feature_vectors, b)
    cost_minimized = False
    
    while  not cost_minimized and trainings < 102:
        w_gradien = w_derivative(w, feature_vectors, b)
        b_gradien = b_derivative(w, feature_vectors, b)

        w = w - (w_gradien * learning_rate)
        b = b - (b_gradien * learning_rate)

        print(f" {w} + {b}")
        cost_of_trainin = cost_calculate(w, feature_vectors, b)
        # time.sleep(2) 
        print(cost_of_trainin)
        trainings = trainings + 1

        def cost_difference(cost, margin):
            if (len(y) > 0):
                return abs(y[-1] - cost) < margin
            else:
                return False
        
        cost_minimized = cost_difference(cost_of_trainin, margin_of_error)
        x.append(trainings)
        y.append(cost_of_trainin)
        
        sc.set_offsets(np.c_[x,y])
        fig.canvas.draw_idle()
        plt.show()
        plt.pause(0.002)

    return (w, b)

results = gradient_descent(w_vector, 0, 1, feature_vectors, 0.0001)

w_vector, b = results
## b = 0.0000000000000000000000001
## w_vector = [-0.09064125, 0.08622709, 0.22288214, 0.01455121, 0.00989126, 0.13730702, 0.1104075, 0.04684105, 0.3067629, 0.20787016, 0.07358903, -0.2094933, 0.02173129, -0.08484939, 0.22741904, -0.08252611, 0.04074504, -0.02848463]
## print(w_vector)


print(f" {((np.dot(w_vector, feature_vectors[28]) + b) * output_standardd) + output_mean} ")