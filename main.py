import random
import numpy
import numpy as np
import pandas
import pandas as pd
import matplotlib.pyplot as plt

# google colab link:
# https://colab.research.google.com/drive/1hknS5vCXFth1Kcj3BpPVSPJlMwJ7_6Gf?usp=sharing


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


df = pd.read_csv('/content/train.csv').iloc[:43500, :]

# Converting Non Numeric Values To Numeric Values By Using one_hot_function
city_rank = {}
one_hot_encode(city_rank, df['City'])
rank = [city_rank[x] for x in df['City']]
city_mean = numpy.mean(rank)
city_var = np.var(rank)
df['City Rank'] = [(x - city_mean) / city_var for x in rank]

state_rank = {}
one_hot_encode(state_rank, df['State'])
rank = [state_rank[x] for x in df['State']]
state_mean = np.mean(rank)
state_var = np.var(rank)
df['State Rank'] = [(x - state_mean) / state_var for x in rank]

vin_rank = {}
one_hot_encode(vin_rank, df['Vin'])
rank = [vin_rank[x] for x in df['Vin']]
vin_mean = numpy.mean(rank)
vin_var = np.var(rank)
df['Vin Rank'] = [(x - vin_mean) / vin_var for x in rank]

make_rank = {}
one_hot_encode(make_rank, df['Make'])
rank = [make_rank[x] for x in df['Make']]
make_mean = numpy.mean(rank)
make_var = np.var(rank)
df['Make Rank'] = [(x - make_mean) / make_var for x in rank]

model_rank = {}
one_hot_encode(model_rank, df['Model'])
rank = [model_rank[x] for x in df['Model']]
model_mean = numpy.mean(rank)
model_var = np.var(rank)
df['Model Rank'] = [(x - model_mean) / model_var for x in rank]

year_mean = np.mean(df['Year'])
year_var = np.var(df['Year'])
df['Year'] = [(x - year_mean) / year_var for x in df['Year']]

mileage_mean = np.mean(df['Mileage'])
mileage_var = np.var(df['Mileage'])
df['Mileage'] = [(x - mileage_mean) / mileage_var for x in df['Mileage']]


columns = ['Year', 'Mileage', 'City Rank', 'State Rank', 'Vin Rank', 'Make Rank', 'Model Rank']
x_vector = df.loc[:, columns].to_numpy()

# Adding A New Column With Value 1, Just To Set The Bias Along Side With W
bias = np.ones(len(df))
x_vector = np.c_[x_vector, bias]
print('x_vector:')
print(x_vector[0])

target = np.array(df.iloc[:,0].values)
# target_mean = np.mean(target)
# target_var = np.var(target)
# target = (target - target_mean) / target_var
print('target:')
print(target)

learning_rate = 0.05
epoch = 1000

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
df = pd.read_csv('/content/train.csv')
df = pd.read_csv('/content/train.csv').iloc[43500:, :]

# Converting Non Numeric Values To Numeric Values By Using rank function
df['City Rank'] = [(one_hot_decode(city_rank, x) - city_mean) / city_var for x in df['City']]
df['State Rank'] = [(one_hot_decode(state_rank, x) - state_mean) / state_var for x in df['State']]
df['Vin Rank'] = [(one_hot_decode(vin_rank, x) - vin_mean) / vin_var for x in df['Vin']]
df['Make Rank'] = [(one_hot_decode(make_rank, x) - make_mean) / make_var for x in df['Make']]
df['Model Rank'] = [(one_hot_decode(model_rank, x) - model_mean) / model_var for x in df['Model']]

df['Year'] = [(x - year_mean) / year_var for x in df['Year']]

df['Mileage'] = [(x - mileage_mean) / mileage_var for x in df['Mileage']]

test_x = df.loc[:, columns].to_numpy()
test_bias = np.ones(len(df))
test_x = np.c_[test_x, test_bias]

df['Predicted Price'] = numpy.dot(test_x, w)
# df['Predicted Price'] = [(x * target_var) + target_mean for x in df['Predicted Price']]

print('Total Predicted Price:')
print(df['Predicted Price'].to_string())

plt.plot([x for x in range(len(df))], np.array(df.iloc[:,0].values), color='b')
plt.plot([x for x in range(len(df))], np.array(df.loc[:,['Predicted Price']].values), color='r')
plt.savefig('My Test On 10 Test date.png')
plt.show()
# print('Target Mean:')
# print(target_mean)
# print('Target Var:')
# print(target_var)
mins = [min(df['Predicted Price']), min(df['Price'])]
maxs = [max(df['Predicted Price']), max(df['Price'])]
plt.plot([min(mins), max(maxs)], [min(mins), max(maxs)])
plt.scatter(df['Predicted Price'], df['Price'])
plt.xlabel('Predicted Price')
plt.ylabel('Price')
plt.savefig('Professional-Result.png')
plt.show()
exit(0)
###################################################################
# df = pd.read_csv('test.csv')
# # just in case that we can see the result, if we get all test.csv file, we get error of merror space due to its huge size
# df = pd.read_csv('/content/train.csv').iloc[43500:, :]

# # Converting Non Numeric Values To Numeric Values By Using rank function
# df['City Rank'] = [one_hot_decode(city_rank, x) for x in df['City']]
# df['State Rank'] = [one_hot_decode(state_rank, x) for x in df['State']]
# df['Vin Rank'] = [one_hot_decode(vin_rank, x) for x in df['Vin']]
# df['Make Rank'] = [one_hot_decode(make_rank, x) for x in df['Make']]
# df['Model Rank'] = [one_hot_decode(model_rank, x) for x in df['Model']]


# df['Year'] = [(x - year_mean) / year_var for x in df['Year']]

# df['Mileage'] = [(x - mileage_mean) / mileage_var for x in df['Mileage']]

# print(df.head().to_string())

# # columns = ['Year', 'Mileage', 'City Rank', 'State Rank', 'Vin Rank', 'Make Rank', 'Model Rank']
# test_x = df.loc[:, columns].to_numpy()
# test_bias = np.ones(len(df))
# test_x = np.c_[test_x, test_bias]

# df['Predicted Price'] = numpy.dot(test_x, w)
# print(df.to_string())
# df.to_csv('predicted Date.csv')
