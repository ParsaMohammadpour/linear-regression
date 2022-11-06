import random

import numpy
import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt


def one_hot_encode(dictionary, df):
    for x in df:
        if x not in dictionary.keys():
            dictionary[x] = len(dictionary)

def one_hot_decode(dictionary, key):
    if key in dictionary.keys():
        return dictionary[key]
    dictionary[key] = len(dictionary)
    return dictionary[key]


# def loss_function(predicted, target):
#     return ((predicted - target) ** 2) / 2
#
# def cost_function(predicts, targets):
#     return sum([loss_function(predicts[x], targets[x]) for x in range(len(targets))]) / len(targets)

def cost_function_simplified(predicts, targets):
    return np.sum((predicts - targets) ** 2) / (2 * len(targets))

def gradian_function(predict, target, w, x_vector):
    return np.dot(predict - target, x_vector) / len(x_vector)
    # diff = predict - target
    # gradian = np.dot(diff, x_vector)
    # return gradian / len(x_vector)

df = pd.read_csv('train.csv').iloc[:1000, :]

# Converting Non Numeric Values To Numeric Values By Using one_hot_function
city_rank = {}
one_hot_encode(city_rank, df['City'])
rank = [city_rank[x] for x in df['City']]
rank_norm = numpy.linalg.norm(rank)
df['City Rank'] = [x / rank_norm for x in rank]

state_rank = {}
one_hot_encode(state_rank, df['State'])
rank = [state_rank[x] for x in df['State']]
rank_norm = numpy.linalg.norm(rank)
df['State Rank'] = [x / rank_norm for x in rank]

vin_rank = {}
one_hot_encode(vin_rank, df['Vin'])
rank = [vin_rank[x] for x in df['Vin']]
rank_norm = numpy.linalg.norm(rank)
df['Vin Rank'] = [x / rank_norm for x in rank]

make_rank = {}
one_hot_encode(make_rank, df['Make'])
rank = [make_rank[x] for x in df['Make']]
rank_norm = numpy.linalg.norm(rank)
df['Make Rank'] = [x / rank_norm for x in rank]

model_rank = {}
one_hot_encode(model_rank, df['Model'])
rank = [model_rank[x] for x in df['Model']]
rank_norm = numpy.linalg.norm(rank)
df['Model Rank'] = [x / rank_norm for x in rank]

columns = ['Year', 'Mileage', 'City Rank', 'State Rank', 'Vin Rank', 'Make Rank', 'Model Rank']
x_vector = df.loc[:, columns].to_numpy()

# Adding A New Column With Value 1, Just To Set The Bias Along Side With W
bias = np.ones(len(df))
x_vector = np.c_[x_vector, bias]
print('x_vector:')
print(x_vector[0])

target = np.array(df.iloc[:,0].values)
print('target:')
print(target)

learning_rate = 0.0000000001
epoch = 100000

# Setting The Bias Number In This Vector Along Side Of Xi
w = [((random.random() * 1)) for x in range(len(columns) + 1)]
w = [((random.random() * 1)) for x in range(len(columns) + 1)]
print('W:')
print(w)


predict = np.dot(x_vector, w)
print('predict:')
print(predict)
cost = cost_function_simplified(predict, target)
print('cost:')
print(cost)

coasts = []
for i in range(epoch):
    gradian = gradian_function(predict, target, w, x_vector)
    w = w - (learning_rate * gradian)
    print('New W:')
    print(w)
    predict = np.dot(x_vector, w)
    # print('Predict:')
    # print(predict)
    cost = cost_function_simplified(predict, target)
    print('cost:')
    print(cost)
    coasts.append(cost)

x = [i for i in range(len(coasts))]
# plt.plot(x, coasts, marker = 'o', mec = 'r', mfc = 'r', linestyle  = "--")
plt.plot(x, coasts)
plt.xlabel('Test Number')
plt.ylabel('Cost Function')
plt.title('Cost Function Result')
plt.savefig("Loss-Function-Plot.png")
plt.show()


###################################################################
##############         My Test          ####################
df = pd.read_csv('train.csv')
df = pd.read_csv('train.csv').iloc[1570:1590, :]

# Converting Non Numeric Values To Numeric Values By Using rank function
df['City Rank'] = [one_hot_decode(city_rank, x) for x in df['City']]
df['State Rank'] = [one_hot_decode(state_rank, x) for x in df['State']]
df['Vin Rank'] = [one_hot_decode(vin_rank, x) for x in df['Vin']]
df['Make Rank'] = [one_hot_decode(make_rank, x) for x in df['Make']]
df['Model Rank'] = [one_hot_decode(model_rank, x) for x in df['Model']]

test_x = df.loc[:, columns].to_numpy()
test_bias = np.ones(len(df))
test_x = np.c_[test_x, test_bias]

df['Predicted Price'] = numpy.dot(test_x, w)

print('Total Predicted Price:')
print(df['Predicted Price'].to_string())

plt.plot([x for x in range(len(df))], np.array(df.iloc[:,0].values), color='b')
plt.plot([x for x in range(len(df))], np.array(df.loc[:,['Predicted Price']].values), color='r')
plt.savefig('My Test On 10 Test date.png')
plt.show()
exit(0)
###################################################################
df = pd.read_csv('test.csv')
# just in case that we can see the result, if we get all test.csv file, we get error of merror space due to its huge size
df = pd.read_csv('train.csv').iloc[:1000, :]

# Converting Non Numeric Values To Numeric Values By Using rank function
df['City Rank'] = [one_hot_decode(city_rank, x) for x in df['City']]
df['State Rank'] = [one_hot_decode(state_rank, x) for x in df['State']]
df['Vin Rank'] = [one_hot_decode(vin_rank, x) for x in df['Vin']]
df['Make Rank'] = [one_hot_decode(make_rank, x) for x in df['Make']]
df['Model Rank'] = [one_hot_decode(model_rank, x) for x in df['Model']]

print(df.head().to_string())

# columns = ['Year', 'Mileage', 'City Rank', 'State Rank', 'Vin Rank', 'Make Rank', 'Model Rank']
test_x = df.loc[:, columns].to_numpy()
test_bias = np.ones(len(df))
test_x = np.c_[test_x, test_bias]

df['Predicted Price'] = numpy.dot(test_x, w)
print(df.to_string())
df.to_csv('predicted Date.csv')