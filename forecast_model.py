from sklearn.model_selection import train_test_split,KFold,cross_val_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
from tensorflow import keras
!pip install seaborn
#!pip install xgboost
from tensorflow.keras import regularizers
from keras.models import Sequential
import xgboost
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#loading first training data and checking any null values is present are not
train=pd.read_csv("solar.csv")
print(train.info())
print(train.isnull())
#droping down of timestamp feature because here in this case it's not useful
train.drop("Timestamp",axis=1,inplace=True)
train.head()
y=train.iloc[:,6:9]
train.drop(["Clearsky DHI","Clearsky GHI","Clearsky DNI"],axis=1,inplace=True)
y_Train=y_train.astype(np.float32)
#spliting data
x_train,x_test,y_Train,y_test=train_test_split(train,y,test_size=0.2,random_state=100)
#scaling down of all features with minmaxscaler and standardscaler
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x_train_scaler=scaler.fit_transform(x_train)
x_test_scaler=scaler.fit_transform(x_test)
#standardscaler
from sklearn.preprocessing import StandardScaler
pipeline=Pipeline([
    ("scaler",StandardScaler)
])
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.fit_transform(x_test)
#stack layers with regularizations and mape error with minimum rate
def solar():
    model=tf.keras.Sequential([
        keras.layers.Dense(100,activation="relu",input_shape=[12],kernel_regularizer=regularizers.l2(0.3)),
        keras.layers.Dense(100,activation="relu",kernel_regularizer=regularizers.l2(0.2)),
        keras.layers.Dense(1,activation="linear",kernel_regularizer=regularizers.l2(0.2))
        
    ])
    model.compile(loss="mean_absolute_percentage_error",optimizer="adam",metrics=["MAE"])
    return model
model=solar()
history=model.fit(x_train_scaled,y_Train,epochs=66,batch_size=66,validation_data=(x_test_scaler,y_test))
#now it's time for visualization model error on train data and validation data
pd.DataFrame(history.history).plot(figsize=(12,6))
plt.show()
#knnregressor
from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor(n_neighbors=20).fit(x_train_scaler,y_train)
pred=knn.predict(x_test_scaler)
print(pred)
knn.score(x_train_scaler,y_train)
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(y_test,pred)
import math
print(math.sqrt(mse))
r2_score=r2_score(y_test,pred)
#xgb regressor
import xgboost as xgb
xgb_regressor=xgb.XGBRegressor().fit(x_train,y_train)
xgb_predictions=xgb_regressor.predict(x_test)
xgb_regressor.score(x_train,y_train)
plt.figure(figsize=(10,6))
plt.plot(pred,label="prediction")
plt.plot(y_test.values,label="actual")
plt.legend()
plt.title("knn regressor")
plt.xlabel("observations")
plt.ylabel("targetvalues")
plt.show()
#with forecasting method VAR
from statsmodels.tsa.api import VAR
model=VAR(train)
results=model.fit()
print(results.summary())
n_periods=len(x_test)
forecast=results.forecast(x_train.values[-results.k_ar:],n_periods)
print(forecast)
mse=np.mean((forecast-x_test.values)**2)
mse
#testing with few test samples
test=pd.read_csv("solar_test.csv")
test.drop(["Clearsky DHI","Clearsky DNI","Clearsky GHI"],axis=1,inplace=True)
import pandas as pd
import numpy as np

# create an example DataFrame with 52560 rows
submission=pd.read_csv("submission.csv")
df = pd.DataFrame(np.random.randn(52560, 5), columns=['A', 'B', 'C', 'D', 'E'])

# create an empty DataFrame with the same number of rows
submission = pd.DataFrame(index=df.index, columns=['Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI'])

# generate some example values for the three columns
pred = np.random.randn(52560, 3)

# assign the values to the appropriate columns in the new DataFrame
submission[['Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI']] = pred

# check that the new DataFrame has the correct number of rows and columns
print(submission.shape)  # should output (52560, 3)
plt.figure(figsize=(10,6))
plt.plot(pred,label="prediction")
plt.plot(y_test.values,label="actual")
plt.legend()
plt.title("knn regressor")
plt.xlabel("observations")
plt.ylabel("targetvalues")
plt.show()
#measuring model bias and uncertainty
bias_wrapped_solar_nn=capsa.HistogramWrapper(

    model,
    num_bins=20,
    queue_size=2000,
    target_hidden_layer=False
)
bias_wrapped_solar_nn.compile(

    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
    loss=tf.keras.losses.MeanSquaredError()
)
loss_historry_bias_wrap=bias_wrapped_solar_nn.fit(x_train_scaled,y_train,epochs=30)
print("done training model with Bias Wrapper#")
bias=bias_wrapped_solar_nn(x_test_scaled)
