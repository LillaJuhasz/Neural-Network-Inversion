import pandas as pd
import numpy as np
from numpy import array
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


ds = pd.read_csv('OnlineNewsPopularity.csv')
ds_copy = ds.drop(columns='url')
#ds_corr = ds.corr()

'''ds_norm= (ds_corr - ds_corr.values.min(0)) / ds_corr.values.ptp(0)
print(ds_norm, file=open('news_norm.txt', 'a'))'''



####################### FEATURE CROSSING #########################


data_channels = ds_copy[['data_channel_is_lifestyle', 'data_channel_is_entertainment',
                        'data_channel_is_bus', 'data_channel_is_socmed', 'data_channel_is_tech',
                        'data_channel_is_world']]
data_channels = list(data_channels.values)
ds_copy.insert(60, 'data_channels', data_channels)

##################################################################

weeks = ds_copy[['weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday',
                 'weekday_is_thursday', 'weekday_is_friday', 'weekday_is_saturday',
                 'weekday_is_sunday']]
weeks = list(weeks.values)
ds_copy.insert(61, "weeks", weeks)



##################### DROP USELESS FEATURES ######################


ds_copy = ds_copy.drop(columns=['timedelta', 'data_channel_is_lifestyle', 'data_channel_is_entertainment',
                        'data_channel_is_bus', 'data_channel_is_socmed', 'data_channel_is_tech',
                        'data_channel_is_world', 'weekday_is_monday', 'weekday_is_tuesday',
                        'weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday',
                        'weekday_is_saturday', 'weekday_is_sunday', 'is_weekend'])

ds_copy = ds_copy.drop(columns=['data_channels','weeks'])

##################################################################

ds_copy = ds_copy[ds_copy.n_unique_tokens < 1]
ds_copy = ds_copy[ds_copy.n_tokens_content != 0]
ds_copy = ds_copy[ds_copy.n_unique_tokens != 0]
ds_copy = ds_copy[ds_copy.average_token_length != 0]
ds_copy = ds_copy[ds_copy.num_hrefs <= 100]
ds_copy = ds_copy[ds_copy.num_self_hrefs <= 10]
ds_copy = ds_copy[ds_copy.num_imgs <= 10]
ds_copy = ds_copy[ds_copy.num_videos <= 2]
ds_copy = ds_copy.reset_index(drop=True)



####################### DISPLAYS AND PLOTS #######################


#print(ds_copy.describe())
#print(ds_copy.describe(), file=open('corr_describe.txt', 'w'))

'''mask = np.zeros_like(ds_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(ds_corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5,
            annot=True, annot_kws={"size": 5}, fmt='1.1f', cbar_kws={"shrink": .5})
plt.show()'''



######################## DATASET SCALING ##########################


scaler = StandardScaler()
ds_copy[:]=scaler.fit_transform(ds_copy[:])

y = ds_copy.get(key='shares').values
X = ds_copy.drop(columns='shares').values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)



########################## TRAINING ###############################


mlp = MLPRegressor(verbose=True)
param_grid = {
    'hidden_layer_sizes': [(10, 20, 40, 20, 10), (30, 90, 180, 90, 30), (100, 400, 100), (1000,2000)],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['lbfgs','sgd', 'adam'],
}
gs = GridSearchCV(mlp, param_grid, cv=2)
gs.fit(X_train, y_train)

###################################################################

print(gs.best_params_)
print(gs.score(X_test, y_test))
y_pred = gs.predict(X_test)

###################################################################

with open('gridSearchResults.txt', 'a') as f:
    print("{} \n \n {} \n  \n Best Estimator: \n {} \n with {}".format(
        param_grid, gs.cv_results_, gs.best_estimator_, gs.best_score_), file=f)
