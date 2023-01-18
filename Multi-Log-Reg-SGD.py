### Multinomial Logistic Regression

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
import math

# Goal: predict disease class from the data. \
# We will be using multinomial logistic regression.
# Dataset can be found at: https://archive.ics.uci.edu/ml/datasets/dermatology

# Pre-processing
# dermatology.csv can be found in the GitHub
derm_data = pd.read_csv("dermatology.csv", delimiter = ',')
derm_data.head()

# We change the column names to make the data interpretable.
colnames = ['erythema', 'scaling', 'definite borders', 
            'itching', 'koeber phenomenon', 'polygonal papules', 'follicular papules', 'oral mucosal involvement', 
            'knee and elbow invovlement', 'scalp involvement', 'family history', 'mealnin incontinence', 
            'eosinophils in the infiltrate', 'PNL infiltrate', 'fibrosis of the papillary dermis', 
            'exocytosis', 'acanthosis', 'hyperkeratosis', 'parakeratosis', 'dlubbing of the rete ridges', 
            'elongation of rete ridges', 'thinning of the suprapapillary epidermis', 'spongifrom pustule', 'munor microabcess', 'focal hypergranulosis',
           'disappearnce of the granular layer', 'vaculoisation and damage of basil layer', 'spongiosis', 'saw-tooth appearnce of retes', 'follicular horn plug',
           'perifollicular parakeratosis', 'inflammatory monoluclear infiltrate', 'band-like infiltrate', 'Age(linear)', 'Classcode']

derm_data.columns = colnames

# Remove all NA's from data
for k in range(derm_data.shape[0]):
    for i in range(derm_data.shape[1]):
        if (derm_data.iloc[k,i] == '?'):
            derm_data.iloc[k,i] = 'nan'
derm_data[['Age(linear)']].fillna(value=derm_data[['Age(linear)']].mean(),inplace=True)

derm_data.head()

# Since logistic regression tries to find a linear boundary between classes in our data, we will first try to visualize our data.

derm_data.describe()

plt.figure(figsize=(12,12))
sns.heatmap(derm_data.corr())

# When we draw a histogram of the spread of the "Classcode" column, we see that there is a much higher frequency of Class 1 Disease (psoriasis) than any other disease.

# Visualization - lots of cases of class 1 disease (psoriasis)
plt.hist(derm_data[['Classcode']])
plt.xlabel('Classcodes')
plt.ylabel('Value Counts')
plt.title('Histogram of Classcodes (Disease Frequency)')

# We split our dataset into our features (34 features) and our target (1 target).

features = derm_data[['erythema', 'scaling', 'definite borders', 
            'itching', 'koeber phenomenon', 'polygonal papules', 'follicular papules', 'oral mucosal involvement', 
            'knee and elbow invovlement', 'scalp involvement', 'family history', 'mealnin incontinence', 
            'eosinophils in the infiltrate', 'PNL infiltrate', 'fibrosis of the papillary dermis', 
            'exocytosis', 'acanthosis', 'hyperkeratosis', 'parakeratosis', 'dlubbing of the rete ridges', 
            'elongation of rete ridges', 'thinning of the suprapapillary epidermis', 'spongifrom pustule', 'munor microabcess', 'focal hypergranulosis',
           'disappearnce of the granular layer', 'vaculoisation and damage of basil layer', 'spongiosis', 'saw-tooth appearnce of retes', 'follicular horn plug',
           'perifollicular parakeratosis', 'inflammatory monoluclear infiltrate', 'band-like infiltrate', 'Age(linear)']]
features = features.to_numpy()
target = derm_data[['Classcode']]
target = target.to_numpy()
target = np.squeeze(target)

features.shape, target.shape

# Now we standardize the data to ensure that our data has a standard deviation of 1.

scaler = StandardScaler().fit(features)
features = scaler.transform(features)

# Successfully standardized the features
for i in range(features.shape[1]):
    print(np.std(features[:,i]))
    
# Now we define our logit function that will take in our features, and our randomly initialized $w_0$ (weight parameter) and $b_0$ (bias parameter) and calculate a logit score for each feature. As a reminder, a logit function (log-odds function) calculates the probability of a certain event occurring in the domain of [0,1]. It is the inverse of the logistic sigmoid function, which predicts probabilities in the range of [0,1].

def logitscores(features, w_0, b_0):
    logitvalues = np.zeros(shape=(features.shape[0],6))
    for i in range(len(logitvalues)):
        # For every row in the features dataframe, take the dot product with the weights and add the bias
        # w^Tx + b => hyperplane
        logit = np.squeeze(np.dot(w_0, features[i]) + np.transpose(b_0))
        logitvalues[i,:] = logit
    return logitvalues
  
# Now we define a softmax function. In multinomial logistic regression, we use a softmax function, which calculates the probabilities of K possible outcomes for a vector of K numbers. This is a generalization of the sigmoid function used in binary logistic regression. 
# The softmax function is: $\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$

def softmax(logitvalues, axis=-1):
    K_prob = np.zeros(logitvalues.shape)
    # We us kw to scale the softmax values to be between 0 and 1
    kw = dict(axis=axis, keepdims = True)
    logitvalues = logitvalues - logitvalues.max(**kw)
    for i in range(logitvalues.shape[0]):
        # Use the definition of the softmax function
        a = np.exp(logitvalues[i])
        b = a/a.sum(**kw)
        K_prob[i] = b
    return K_prob
  
 # Now we define our Logisitc Regression function. We calculate the logit scores, the probability matrix (the probability of each data point (person) being classified in each class (from 1-6), and classify each data point (person) into the class with the highest probability.

def TrainingLogisticRegression(X_train, w_0, b_0):
    logits = logitscores(X_train, w_0, b_0)
    K_prob = softmax(logits)
    K_prob = np.nan_to_num(K_prob,10**-10)
    predict = np.zeros(shape=X_train.shape[0])
    for i in range(K_prob.shape[0]):
        predict[i] = np.argmax(K_prob[i])
    return K_prob, predict
  
# We initialize random weights and biases to test our functions. 
 
w_0 = np.random.randn(6, 34)
b_0 = np.random.randn(6,1)

K_prob, predict = TrainingLogisticRegression(features, w_0, b_0)
print(K_prob.shape)
print(predict.shape)

def model_accuracy(predict, target):
    return 100.0 * np.sum(predict == target)/len(target)

model_accuracy(predict, target)

# The model accuracy is pretty low since this it is generic, so we will improve our model by optimizing hyperparameters.

### Optimizing our model

# We split our data into training and testing data.

# running train_test_split for our dataset
train_features, test_features, train_target, test_target = train_test_split(features,target,test_size = 0.2)

train_features.shape, test_features.shape, train_target.shape, test_target.shape

# We scale the data to ensure a standard deviation of 1.

scaler = StandardScaler().fit(train_features)
train_features = scaler.transform(train_features)
scaler = StandardScaler().fit(test_features)
test_features = scaler.transform(test_features)

for i in range(train_features.shape[1]):
    print(np.std(train_features[:,i]))
for i in range(test_features.shape[1]):
    print(np.std(test_features[:,i]))
    
# In multinomial logistic regression, maximizing the likelihood of the parameters is equivalent to minimizing the cross-entropy loss. We will define our cross-entropy loss function that we will later minimize using an algorithm. 

# As a reminder, cross-entropy loss is defined as:

# $$J(\theta) = -[\sum_{i=1}^m y^{(i)} \log h_{\theta} (x^{(i)}) + (1-y^{(i)} \log (1- h_{\theta}(x^{(i)}))]$$

# where $(x^{(i)}, y^{(i)})$ correspond to a data point from our training set and $h_{\theta}$ is the probability of our parameter set being equal to $\theta$.

def CELoss(K_prob, target):
    target = target-1
    total_loss = 0
    for i in target:
        total_loss += -np.log(K_prob[i])
    avg_loss = total_loss/K_prob.shape[0]
    return np.average(avg_loss)
  
# We use Stochastic Gradient Descent to minimize our cross-entropy loss. We calculate the probability matrix (the probability of each data point (person) being classified in each class (from 1-6)) and the prediction vector (the predicted class of each data point (person) based on the highest probability). We then find the gradient of our $w_0$ weight vector and $b_0$ bias vector and implement the algorithm:

# $$w_{k+1} = w_k - (\alpha * \nabla(f(w_k)))$$
# $$b_{k+1} = b_k - (\alpha * \nabla(f(b_k)))$$

# for $i = 1,...,\text{K_max}$ where $\text{K_max}$ is the maximum number of iterations.

def SGD(alpha, K_max, target, features, w_0, b_0):

    loss_function = np.zeros(features.shape[0])
 
    for i in range(K_max):
        prob,predict = TrainingLogisticRegression(features, w_0, b_0)
        loss_function[i] = CELoss(prob, target)
        prob[np.arange(features.shape[0]),target] -= 1
        
        grad_weight = np.dot(prob.T,features)
        grad_biases = np.matrix(np.sum(prob, axis = 0)).T
        
        w_0 -= (alpha * grad_weight)
        b_0 -= (alpha * grad_biases)
        
    return w_0,b_0,loss_function
  
# We initialize random $w_0$ and $b_0$ and run our SGD algorithm.

w_0 = np.random.randn(6, 34)
b_0 = np.random.randn(6,1)

w_star, b_star, loss_function = SGD(0.1, 10, train_target-1, train_features, w_0, b_0)

w_star = np.nan_to_num(w_star,0)

test_prob, test_pred = TrainingLogisticRegression(test_features, w_star, b_star)
test_pred = test_pred + 1

model_accuracy(test_pred, test_target)

# --------------------------------------

# We can compare the accuracy of our model with the built-in LogisticRegression function in sklearn.

train_features = train_features.astype('float')
test_features = test_features.astype('float')

train_features = np.nan_to_num(train_features,0)
test_features = np.nan_to_num(test_features,0)

# We use the LogisticRegression function in sklearn, with argument multiclass = 'multinomial'.

model = LogisticRegression(solver = 'newton-cg', multi_class = 'multinomial')
model.fit(train_features, train_target)
class_pred = model.predict(train_features)
score = model.score(test_features, np.squeeze(test_target))
print(score)

# The sklearn accuracy score is very similar to our accuracy score.

# We also do 5-fold cross validation and take the average to find the average model accuracy of our five repetitions.

cross_val = LogisticRegression(solver='newton-cg', multi_class='multinomial')
scores = cross_val_score(cross_val, train_features, train_target, cv=5)
scores

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# The accuracy is very high, similar to our model.

# This was an implementation of multinomial logistic regression on a dataset to classify people into six different disease classes based on 34 features using the publicly available dataset from https://archive.ics.uci.edu/ml/datasets/dermatology.

