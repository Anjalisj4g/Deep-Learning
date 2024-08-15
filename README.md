# Deep Learning

**Objective:**

You are required to model the progression of diabetes using the available independent variables. This model will help healthcare professionals understand how different factors influence the progression of diabetes and potentially aid in designing better treatment plans and preventive measures. The model will provide insights into the dynamics of diabetes progression in patients.

Installed required libraries
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential

from keras.layers import Dense

from keras import utils

from keras.optimizers import Adam

**Dataset used : Diabetes dataset**

from sklearn.datasets import load_diabetes

Dataset is checked for any null values. There are no null values.

**Visualization**

Histplot is used to show the distribution of target variable.

![image](https://github.com/user-attachments/assets/08d943e8-99a1-4bf2-9433-b8a7d902d62b)

Histplot is used to show the distribution of features.

![image](https://github.com/user-attachments/assets/3a28046e-5b51-49c4-a9d0-c67fa9778b45)

Pair plot is drawn to find the distribution of features.

Heatmap is used to find the relationship between features and target variables.

![image](https://github.com/user-attachments/assets/b1a3724d-8edc-44dc-84b5-a0fce311f7af)

**Created a Deep Neural Network model**

model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))

model.add(Dense(32, activation='relu'))

model.add(Dense(1)) 

**Evaluated the model :**

Mean Absolute Error : 44.37

**Created an ANN with 50 hidden layers**

input_dim = X_train.shape[1]  # Automatically match to the dataset's number of features

num_hidden_layers = 50

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=input_dim))

for _ in range(num_hidden_layers - 1):

    model.add(Dense(units=64, activation='relu'))

model.add(Dense(units=1))  # Output layer with 1 neuron for regression

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_absolute_error'])

model.fit(X_train, y_train, epochs=100)

**Evaluated the model :**

Mean absolute error is 56.27

**Improved model with different architectures, activation functions or hyperparameters**

* Mean absolute error has decreased from 56.27 to 41.84 using FeedForward Neural Network.
* Mean absolute error has decreased from 56.27 to 41.67 using Activation function.
* Mean absolute error has decreased from 56.27 to 44.5 using different hyperparameters.





