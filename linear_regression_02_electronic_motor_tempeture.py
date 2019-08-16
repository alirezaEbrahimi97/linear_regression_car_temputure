import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

def devide_data(src, frac_value):
    print("started deviding data for train and test")
    test_data = src.sample(frac = frac_value)
    merged = src.merge(test_data, how='left', indicator=True)
    train_data = merged[merged['_merge'] == 'left_only']
    print("deviding compelete")
    return test_data, train_data

def make_x_y(src):
    print("started making x and y")
    heads = src.columns.to_list()
    heads.remove('ambient')
    heads.remove('profile_id')
    heads.remove('coolant')
    heads.remove('stator_tooth')
    heads.remove('stator_winding')
    if '_merge' in heads:
        heads.remove('_merge')    
    X = src[heads]
    Y = src['ambient']
    print("compelete")
    return X, Y

def one_hot(column_name, df):
    one_hot_data = pd.get_dummies(df[column_name], prefix=[column_name])
    return  pd.concat([df, one_hot_data], axis=1)

df = pd.read_csv("electric-motor-temperature.zip")
df = one_hot('profile_id', df)
df_test, df_train = devide_data(df, 0.2)
x_test, y_test = make_x_y(df_test)
x_train, y_train = make_x_y(df_train)

model = linear_model.LinearRegression()
model.fit(x_train.to_numpy(), y_train.to_numpy())
y_predict = model.predict(x_test.to_numpy())
plt.plot(y_test.to_numpy(), label='test')
plt.plot(y_predict, label='predict')
plt.legend()
plt.show()

y_predict_np = np.array(y_predict)
dis = y_test.to_numpy() - y_predict_np
plt.plot(np.absolute(dis))
plt.show()
print("mean of ablsolute_distance: ", np.absolute(dis).sum() / dis.shape[0])

print("max and min of ambient", df['ambient'].max(), df['ambient'].min())