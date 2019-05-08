import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import inversion


dataset = pd.read_csv('OnlineNewsPopularity.csv')
dataset_copy = dataset.drop(columns=['url', 'timedelta'])

dataset_copy.describe()

dataset_copy = dataset_copy[dataset_copy.n_tokens_content != 0]
dataset_copy = dataset_copy[dataset_copy.n_unique_tokens < 1]
dataset_copy = dataset_copy[dataset_copy.average_token_length != 0]

dataset_copy = dataset_copy[dataset_copy.num_hrefs <= 100]
dataset_copy = dataset_copy[dataset_copy.num_self_hrefs <= 10]
dataset_copy = dataset_copy[dataset_copy.num_imgs <= 10]
dataset_copy = dataset_copy[dataset_copy.num_videos <= 2]
dataset_copy = dataset_copy.reset_index(drop=True)


scaler = StandardScaler()
dataset_copy[:] = scaler.fit_transform(dataset_copy[:])

y = dataset_copy.pop('shares')
X = dataset_copy


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)


mlp = MLPRegressor()
param_grid = {
    'hidden_layer_sizes': [(100,100), (100,200,500), (50,150,50), (20,80,80,20),
                           (30,90,180,90,30), [200,500,200], (100,200,500,200,100),
                           (1000,2000)],
    'activation': ['relu', 'logistic', 'tanh'],
    'solver': ['lbfgs', 'adam'],
    'alpha': [0.03, 0.01, 0.003, 0.001],
    'learning_rate_init': [0.03, 0.01, 0.003, 0.001],
    'learning_rate': ['adaptive'],
}
gs = GridSearchCV(mlp, param_grid, cv=2, n_jobs=-1)
gs.fit(X_train, y_train)

with open('gridSearchResults.txt', 'a') as f:print(
    "{} \n \n {} \n  \n Best Estimator: \n {} \n with "
    "{}".format(param_grid, gs.cv_results_, gs.best_estimator_,
                gs.best_score_), file=f)

regressor = gs.best_estimator_
print(regressor, 'score: ', gs.best_score_, sep='\n ', file=open('TrainingResult.txt', 'w'))

y_pred = regressor.predict(X_test)

plt.plot(X_test, y_test, 'o', color='orange')
plt.plot(X_test, y_pred, 'o', color='blue')
plt.savefig("./" + 'plot.pdf')
plt.show()

inversionResults = pd.DataFrame(columns=['accuracy_percent'])

for i, value in enumerate(y_test):
    desired_output = [value]
    guessedInput = inversion.invert(regressor, desired_output, pd.DataFrame(X_test).columns.size,
                                    gs.best_params_['learning_rate_init'])
    guessedInput = pd.DataFrame(guessedInput).T

    accuracy = 1 - abs((regressor.predict(guessedInput) - desired_output) / (y_test.max() - y_test.min()))
    inversionResults = inversionResults.append(
        {'accuracy_percent': accuracy[0]*100}, ignore_index=True)

    print('guessed input vector in X_test: ', np.array(guessedInput),
          'predicted output vector for y_test: ', regressor.predict(guessedInput),
          'desired output value in y_test: ', desired_output,
          'error: ', regressor.predict(guessedInput) - desired_output,
          'accuracy percent: ', accuracy[0]*100, '\n',
          sep='\n ', file=open('InversionResults.txt', 'a'))

print('summarization: ', inversionResults.describe(), sep='\n ', file=open('InversionResults.txt', 'a'))
