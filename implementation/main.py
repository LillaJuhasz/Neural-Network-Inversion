import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor


dataset = pd.read_csv('OnlineNewsPopularity.csv')
dataset_copy = dataset.drop(columns=['url','timedelta'])

#dataset_copy.describe()


##################### DROP USELESS FEATURES ######################


dataset_copy = dataset_copy[dataset_copy.n_tokens_content != 0]
dataset_copy = dataset_copy[dataset_copy.n_unique_tokens < 1]
dataset_copy = dataset_copy[dataset_copy.average_token_length != 0]

dataset_copy = dataset_copy[dataset_copy.num_hrefs <= 100]
dataset_copy = dataset_copy[dataset_copy.num_self_hrefs <= 10]
dataset_copy = dataset_copy[dataset_copy.num_imgs <= 10]
dataset_copy = dataset_copy[dataset_copy.num_videos <= 2]
dataset_copy = dataset_copy.reset_index(drop=True)



######################## DATASET SCALING ##########################


scaler = StandardScaler()
dataset_copy[:] = scaler.fit_transform(dataset_copy[:])

y = dataset_copy.get(key='shares').values
X = dataset_copy.drop(columns='shares').values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)



########################## TRAINING ###############################


mlp = MLPRegressor()
param_grid = {
    'hidden_layer_sizes': [(30, 90, 180, 90, 30)],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['lbfgs','sgd', 'adam'],
    'alpha': [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001],
    'learning_rate_init': [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001],
    'learning_rate': ['adaptive'],
}
gs = GridSearchCV(mlp, param_grid, cv=2)
gs.fit(X_train, y_train)
y_pred = gs.predict(X_test)

###################################################################

print(gs.best_params_)
print(gs.score(X_test, y_test))
print('shares', gs.best_params_, gs.score(X_test,y_test), sep='; ',
      file=open('InversionResults.txt', 'a'))

with open('gridSearchResults.txt', 'a') as f:print(
    "{} \n \n {} \n  \n Best Estimator: \n {} \n with "
    "{}".format(param_grid, gs.cv_results_, gs.best_estimator_,
                gs.best_score_), file=f)

###################################################################

plt.plot(X_test, y_test, 'o', color='blue')
plt.plot(X_test, y_pred, 'o', color='orange')
plt.savefig('plots/' + 'shares_prediction.pdf')
plt.show()


############################# INVERSION ###########################

X = pd.DataFrame(X)
X.columns = dataset_copy.drop(columns='shares').columns

for X_value in X:
    X_inverse = X.drop(columns=X_value).values
    X_inverse_help = pd.DataFrame(y_train)
    y_pred = pd.DataFrame(y_pred)

    for y_pred_value in y_pred.values:
        y_pred_value = pd.Series(y_pred_value)
        X_inverse_help = X_inverse_help.append(y_pred_value, ignore_index=True)

    X_inverse = pd.DataFrame(X_inverse)
    X_inverse.insert(0,'shares',X_inverse_help)
    X_inverse = X_inverse.values

    y_inverse = X.get(key=X_value).values


    X_inverse_train, X_inverse_test, y_inverse_train, y_inverse_test = \
        train_test_split(X_inverse, y_inverse, test_size=0.3, random_state=0)

    gs.fit(X_inverse_train, y_inverse_train)
    y_inverse_pred = gs.predict(X_inverse_test)

    print(gs.best_params_)
    print(gs.score(X_inverse_test, y_inverse_test))
    print(X_value, gs.best_params_, gs.score(X_inverse_test,y_inverse_test), sep='; ', file=open('InversionResults.txt', 'a'))

    plt.plot(X_inverse_test, y_inverse_test, 'o', color='blue')
    plt.plot(X_inverse_test, y_inverse_pred, 'o', color='orange')
    plt.savefig('plots/' + str(X_value) + '_prediction.pdf')
    plt.show()
