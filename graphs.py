from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("housepricez.csv")

feature_vectors = df[[
    "Bedrooms", "Bathrooms", "Sqft_living", "Grade", "Sqft_above", "Sqft_basement",
    "Long", "Sqft_living15"
]].to_numpy()
output_vectors = df[['Price']].to_numpy()

plt.scatter(output_vectors, np.zeros_like(output_vectors), s=5)
plt.xlim(0, 10000000)

plt.show()

for i in range(0, len(feature_vectors) - 1):
    plt.scatter(feature_vectors[:, i], output_vectors, s=5)
    plt.show()