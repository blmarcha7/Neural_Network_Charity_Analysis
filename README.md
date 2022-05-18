# Neural_Network_Charity_Analysis

## Overview of the Analysis
With our knowledge of machine learning and neural networks, the goal of this analysis was to use the features from the Alphabet Soup data set to help Beks create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

## Results

### Data Preprocessing
Below are the variables which were our features for the model:
-Application_Type 
-Classification 
Below was the target variable:
-Is_Successful
The Variables that were not relevant/important that were removed:
-EIN 
-Name

### Compiling, Training, and Evaluating the Model
Below is the model that gave me the my best accuracy score:

number_input_features = len(X_train[0])
hidden_nodes_layer1 = 80
hidden_nodes_layer2 = 30
hidden_nodes_layer3 = 10
nn = tf.keras.models.Sequential()

First hidden layer
nn.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu")
)

Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

Third hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer3, activation="relu"))

Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

This got me to achieve a 55.4% accuracy, which was below the target 75% accuracy.

I changed the number of nodes, the number of hidden layers, and also changed the activation function of one of my layers for one attempt.

## Summary 
Overall my model didn't do the best job for predicting if the applicants would be successful if funded by Alphabet Soup. I was hoping to get a higher accuracy and I think to achieve that I would have needed to preprocess the data better and clean it up to improve accuracy. I also think I could have adjusted the binning as well.
