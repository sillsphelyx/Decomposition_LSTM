#!/usr/bin/env python
# coding: utf-8

# Importing all necessary packages
import os
#wavelet transform function denoising 
import pywt
from statsmodels.robust import mad
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pylab as pltt
from math import sqrt
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
####################################################################################
#Decomposition
import ewtpy
from PyEMD import EMD, EEMD, CEEMDAN #CEEMD
##################################################################################
import sys
import sklearn as sk
import tensorflow.keras
import tensorflow as tf
print(f"Tensor Flow Version: {tf.__version__}")
#print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")
#####################################################################################
# For correlations 
from sklearn import linear_model
from scipy.stats import gaussian_kde
import statsmodels.api as sm
########################################################################################
# Let's load the required libs.
# We'll be using the Tensorflow backend (default).
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Activation, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
import keras_tuner as kt
from keras.models import load_model
import warnings
import time
warnings.filterwarnings("ignore")
#plt.style.use(['science','notebook'])
#import pandas as pd
#pd.set_option('precision', 2)


def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")
    
    # Set the font to be serif, rather than sans
    sns.set(font='serif')
    
    # Make the background white, and specify the
    # specific font family
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })

# helper function for getting country's cases, deaths, and recoveries        
def get_country_info(dataframe,country_name,col_name = 'Cases'):
    """This helper function is for obtaining univariate time series of (cases,deaths or recoveries) for a
    desired country and returning in the desired format for LSTM model building.\
    Input: -DataFrame containing the desired time series (Cases,deaths or recoveries) obtained from JH CSSEGIS
            containing covid-19 statistics for various countries.
           -Desired country name.
           -Column Name for the output table. eg cou
    Output: DataFrame with index as datetime(days since start of pandemic) and the respective recorded 
            statistics"""
    
    dates = dataframe.columns[4:]
    covid_stats = []
        
    for i in dates:
        covid_stats.append(dataframe[dataframe['Country/Region']==country_name][i].sum())
        #country_deaths.append(deaths_df[deaths_df['Country/Region']==country_name][i].sum())
#         country_recoveries.append(recoveries_df[recoveries_df['Country/Region']==country_name][i].sum())
    dates = pd.to_datetime(dates)
    data = np.array(covid_stats)#, country_deaths))
    transformed_data = pd.DataFrame(data = data, index=dates,columns=[col_name])
    return transformed_data
  


# Wavelet denoising
#adjustable parameters
def wave(wData,wavelet = 'db8',padding = 'smooth',dlevel = 1,thresh=3):    
    """EWT based function for performing EWT-Denoising of univariate timeseries data
       Input: 1D sequence (timeseries data)
       Output: Denoised sequence (timeseries)
       The model has other adjustable parameters including,
       wavelet, padding, dlevel, and thresh""" 
    if len(wData)%2!=0:
        wData = wData.iloc[1:]
    else:
        wData = wData
    newWL = wData.copy() 
    for i in range(0,wData.shape[1]):
        nblck = wData[wData.columns[i]].copy()
        noisy_coefs = pywt.wavedec(nblck, wavelet, level=dlevel, mode=padding)
        sigma = mad(noisy_coefs[-1]).copy()
        uthresh = sigma*thresh
        denoised = noisy_coefs[:]
        denoised[1:] = (pywt.threshold(i, uthresh, 'soft') for i in denoised[1:]) # smoothing
        newWL[wData.columns[i]] = pywt.waverec(denoised, wavelet, mode=padding).copy()
    return newWL


# EEMD Decomposition.
def EEMD_decomp(dataframe):
    """Function for decomposing time series using the EEMD approach.
       Input: pandas dataframe with one column containing the series that you wish to decompose
       Output: a numpy array of shape (number of IMF's, length of series)"""
    t = np.linspace(0, 1, dataframe.shape[0]) # setting time axis in the range 0-1
    s = dataframe[dataframe.columns[0]].values # timesereis
    #Decomposition (Decouple)
    emd = EEMD()
    IMFs = emd.emd(s,t)
    return IMFs


# EEMD Decomposition.
def CEEMDAN_decomp(dataframe):
    """Function for decomposing time series using the EEMD technique."""
    t = np.linspace(0, 1, dataframe.shape[0]) # setting time axis in the range 0-1
    s = dataframe[dataframe.columns[0]].values # timesereis
    #Decomposition (Decouple)
    ceemdan = CEEMDAN()
    c_imfs = ceemdan(s)
    c_imfs = ceemdan.ceemdan(s)
    return c_imfs



# Empirical Wavelet decompostions
def EWT_decomp(dataframe,N=12):
    ewt,  mfb ,boundaries = ewtpy.EWT1D(dataframe[dataframe.columns[0]].values,N=N)
    return ewt.T



def IMF_plot(IMFs,title=''): 
    if IMFs.shape[0] < IMFs.shape[1]:
        columns = [f'IMF{i+1}' for i in range(IMFs.shape[0])]
        frame = pd.DataFrame(data=IMFs.T,columns=columns)
    else:
        columns = [f'IMF{i+1}' for i in range(IMFs.shape[1])]
        frame = pd.DataFrame(data=IMFs,columns=columns)
    set_style()
    plt.figure(figsize=(12, 6))
    frame[columns].plot(subplots=True)
    plt.suptitle(title,size=20)
    plt.xticks(rotation=90)
    plt.savefig(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/{title}.png',facecolor = 'white',transparent = False, bbox_inches="tight")
    plt.show()



def IMF_pl(IMFs): 
    if IMFs.shape[0] < IMFs.shape[1]:
        columns = [f'IMF{i+1}' for i in range(IMFs.shape[0])]
        frame = pd.DataFrame(data=IMFs.T,columns=columns)
    else:
        columns = [f'IMF{i+1}' for i in range(IMFs.shape[1])]
        frame = pd.DataFrame(data=IMFs,columns=columns)
    return frame




def train_test(timeseries,train_size_proportion = 0.7):
    """ train test splitting function. Splits the function according to the given train_size proportions
        Inputs: The series to be split into training and testing sets and the proportion you want for the 
                training set
        Outputs: Two arrays, which are respectively the training and testing sets."""
    train_size = int(len(timeseries) * train_size_proportion)
    test_size = len(timeseries) - train_size
    train, test = timeseries[0:train_size], timeseries[train_size:]
    return (train, test)
    #print("Number of entries (training set, test set): " + str((len(train), len(test))))
#Preprocessing functions
# Scalling function
def scale_datasets_tr(x_train):
    """
      MinmaxScaler. Scaling between [0,1]
    """
    Minmax_scaler = MinMaxScaler()
    x_train_scaled = Minmax_scaler.fit_transform(x_train)      
    #x_test_scaled = standard_scaler.transform(x_test)
    return x_train_scaled#, x_test_scaled

# Data preparation using the timeseries generator
# Generating the time series into supervised learning problem
#Transforming data
def gen_prep(train,test,window_size = 1,batch_size = 1,num_features = 1):
    """Preparing the training and test sets for training and testing DNN
       using the time series generator function """
    train_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(train,train,length=window_size,sampling_rate=1,batch_size=batch_size)
    test_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(test,test,length=window_size,sampling_rate=1,batch_size=batch_size)
    return train_generator,test_generator
################# Optimization rescheme #######################################
#TUNER
# Defining over another set of hyperparameter
# In this model we only optimize the number of LSTM units and learning_rate.
# We shall experiment with other additional hyper parameter later

def build_model_lstm(hp):
    """Lstm function coming together"""
    model = tf.keras.Sequential()  
  # Tune the number of units in the LSTM cell
  # Choose an optimal value between 32-512
    units1 = hp.Int('units1', min_value=224, max_value=512, step=32)
    #units2 = hp.Int('units2', min_value=1, max_value=5, step =1)
    #hp_recurrent_dropout = hp.Choice('recurrent_dropout',values = [0.0,0.1]) #recurrent_dropout
    #hp_dropout= hp.Choice('dropout',values = [0.0,0.1]),dropout = hp_dropout, recurrent_dropout = hp_recurrent_dropout
    hp_activation = hp.Choice('activation',["sigmoid", "relu"])
    model.add(LSTM(units=units1,activation= hp_activation,input_shape = train_generator[0][0].shape[1:]))
    #model.add(LSTM(units = 16,activation = "sigmoid"))
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    model.compile(loss="mean_squared_error")
  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4])

    model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
      loss="mean_squared_error",
      metrics=["mean_squared_error"]
  )

    return model

######################### Running the optimization rescheme on data ################################
#tuner = kt.BayesianOptimization(build_model_lstm5,objective="mean_squared_error",max_trials=20,
 #   directory='keras_tuner_dir', project_name='keras_tuner_d61',overwrite = True)

#####################################################################################################

# Model defining function. "This is because we want to iteratively train the models,
# one for each IMF"
def fit_model(train_generator, validation_generator, window_size = 1, number_features=1,epochs=10):
    model = keras.Sequential()
    
    model.add(LSTM(4, 
                   input_shape = (window_size, number_features)))
    model.add(Dense(1))
    model.compile(loss = "mean_squared_error", 
                  optimizer = "adam",
                  metrics=[tf.metrics.MeanAbsoluteError()])
    history = model.fit(train_generator, epochs=epochs,
                            validation_data=validation_generator,
                             shuffle=False)
    
    return model,history

def IMFs_plotter(IMFs,decomposition_method = ''):
    frame = IMF_pl(IMFs)
    fig, axes = plt.subplots(nrows= min(frame.shape), figsize=(15, 10))
    plt.subplots_adjust(hspace=0.35)
    for i, ax in enumerate(axes):
        col_scheme=np.where(frame[f'IMF{i+1}']>0, 'b','r')
        #ax.xaxis.set_major_formatter(mdates.DateFormatter('%M'))
        ax.bar(frame.index, frame[f'IMF{i+1}'], width=2, align='center', color=col_scheme)
        ax.axhline(y=0, color='k')
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.legend(loc='upper right', fontsize=12)
        #ax.set_yticks(range(-10,10))    
        ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
        ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
        ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
        ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')    
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(16)  
        if i+1 < frame.shape[1]:
            ax.set_ylabel(f'IMF{i+1}', fontsize=15)
        else:
            ax.set_ylabel(f'Residual', fontsize=15)
        ax.grid(True)
    plt.savefig(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/{decomposition_method}_IMFs.png')
#### Plotting functions######
def train_loss_plot(df, title = 'Train-Validation Loss'):
    plt.figure(figsize=(12, 6))
    plt.plot(df['loss'], label='Train', linestyle='dashed')
    plt.plot(df['val_loss'], label='Validation')
    plt.xlabel('Epochs',size=20)
    plt.ylabel('MSE', size =20)
    plt.title(title,size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.legend()
    plt.savefig(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/{title}.png',facecolor = 'white',transparent = False, bbox_inches="tight")
    plt.show()



def direc_sim(true,predict):
    data = np.concatenate((true.reshape(-1,1),predict.reshape(-1,1)),axis=1)
    columns = ['True','Predicted']
    true_pred_df = pd.DataFrame(data=data,columns=columns)
    differences = true_pred_df.diff(periods=1).dropna()
    differences['product'] = differences[differences.columns[0]]*differences[differences.columns[1]]
    return len(differences[differences['product']>0])/len(differences)



# Making prediction. Use test_generator series
def prediction(model,X):
    return model.predict(X)


# Printing target values from the respective generators
def y_from_gen(generator):
    allArrays = np.array([])
    for i in range(len(generator)):
        x, y = generator[i]
        y = y.flatten()
        allArrays = np.concatenate([allArrays, y])
    return allArrays.reshape(-1,1)


# ## Linear model for assessing prediction accuracy

def linear_regression_fit(predicted,true):
    X=predicted
    Y=true
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    predictions = model.predict(X) 
    print_model = model.summary()
    #model.rsquared
    return model


def fitted_plot(model,predicted,true, title = 'IMF1 Actual-Prediction Linear fit'):
    set_style()
    plt.figure(figsize=(12, 6))
    predicted_for_fitting = sm.add_constant(predicted)
    regression_line = model.predict(predicted_for_fitting)
    # Plot the graph
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(predicted,true, label='Data Points', s=75, cmap='viridis')
    plt.plot(predicted, regression_line, label='Best Fit Line', color='black', linewidth=3)
    plt.title(title,size=20)
    plt.xlabel('Predicted values',size = 20)
    plt.ylabel('Actual values',size =20)
    plt.xticks(rotation=90,size=20)
    plt.yticks(size=20)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    #plt.show()
    r2s = str(round(model.rsquared,3))
    ax.text(min(predicted), (0.6*max(true)), '$r^2$ = ' + r2s)
    plt.legend()
    ax.grid(True)  
    plt.savefig(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/{title}.png',facecolor = 'white',transparent = False, bbox_inches="tight")
    plt.show()


def fit_plot_imputations(actual,imputed,title=''):
    lin_model = linear_regression_fit(imputed,actual)
    fitted_plot(lin_model,imputed,actual,title=title)


# Ploting predictions vs actuals
def pred_act_plot(true,predicted,title = 'True and Predicted'):
    plt.figure(figsize=(12, 6))
    set_style()
    plt.plot(true[predicted.shape[0]*-1:], color='blue' ,label='True')
    plt.plot(predicted, color='red' ,label='Predicted')
    plt.title(title,size=20)
    plt.legend()
    plt.ylabel('Number of Cases', size=20)
    plt.xlabel('Date',size=20)
    plt.xticks(rotation=45,size=20)
    plt.yticks(size=20)
    plt.savefig(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results{title}.png',facecolor = 'white',transparent = False, bbox_inches="tight")
    plt.show()


# # Bringing all functions together


# Computing evaluation metrics
def evaluation(true,predicted,model_name = ''):
    score_RMSE = math.sqrt(mean_squared_error(true, predicted))
    DS = direc_sim(true, predicted)
    linear_model = linear_regression_fit(predicted,true) # pred-true prediction assessment
    r_squared = round(linear_model.rsquared,2)
    results =  np.array([score_RMSE,DS,r_squared])
    columns = [model_name]
    dataframe = pd.DataFrame(data = results,columns = columns ,index=['RMSE','DS','r_squared'])
    return dataframe
#Splitter
def X_splitter(X,y,train_test_prop = 0.7):
    """ This is the train test splitter for time series that has already been tranformed into 
        X,y format"""
    train_len = int((X.shape[0])*train_test_prop)    
    X_train, X_test = X[:train_len,:], X[train_len:,:]
    y_train, y_test = y[:train_len], y[train_len:]
    return X_train, X_test, y_train, y_test

#X,y from generator
def X_y_from_gen(generator):
    x_init, y_init = generator[0]
    allarrays = np.zeros((len(generator),x_init.shape[1]))
    allys = np.zeros(len(generator))
    for i in range(len(generator)):
        x, y = generator[i]
        allarrays[i,:] = x.flatten()
        allys[i] = y.flatten()
        #allarrays = allarrays.append(x)
    return allarrays, allys
#gen_prep_init
def gen_prep_init(series_init,window_size=3,batch_size = 1,num_features = 1):
    """Preparing the training and test sets for training and testing DNN
       using the time series generator function """
    train_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(series_init,series_init,length=window_size,sampling_rate=1,batch_size=batch_size)
    return train_generator
# gen_prep_2
def gen_prep_2(X_scaled,y_scaled,batch_size = 1,num_features = 1):
    """Preparing the training and test sets for training and testing DNN
       using the time series generator function """
    train_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(X_scaled,y_scaled,length=1,sampling_rate=1,batch_size=batch_size)
    return train_generator #,test_generator

# Complete model 
def complete_model(series,window_size = 1,train_size_proportion = 0.7,train_val_proportion = 0.7, batch_size=15,epochs = 10,tuner_epochs = 3,max_trials=2, Waveletdenoise = True,
                   Decompose=True,decomp_approach = 'EEMD',max_imf=4,saving = True, model_name="WD_EEMD_LSTM"):
    if (type(series) == np.ndarray) or (type(series) == list):
        series = pd.DataFrame(data = series)
    else:
        series = series
    if Waveletdenoise == True:
        wavelet_denoised_series = wave(series,wavelet = 'db8',padding = 'smooth',dlevel = 1,thresh=3) #denoising
    else:
        wavelet_denoised_series = series  
    if (Decompose == True) and (decomp_approach == 'EEMD'):
        IMFs =  EEMD_decomp(wavelet_denoised_series,max_imf=max_imf) # decomposing with EEMD
    elif (Decompose == True) and (decomp_approach == 'CEEMDAN'):
        IMFs =  CEEMDAN_decomp(wavelet_denoised_series)
    elif (Decompose == True) and (decomp_approach == 'EWT'):
        IMFs =  EWT_decomp(wavelet_denoised_series)    
    else:
        IMFs = np.array([series.values.flatten()])
        
    #EEMD_based_IMFs =  EEMD_decomp(wavelet_denoised_series) # decomposing with EEMD
    results = []
    index = []
    prep_tegen = gen_prep_init(series.values,window_size=window_size)
    train_prediction_total = np.zeros((int(train_size_proportion*len(prep_tegen))))
    #train_prediction_total = np.zeros(len(train_test(IMFs[0].reshape(-1,1),train_size_proportion = train_size_proportion )[0])- window_size)
    test_prediction_total = np.zeros((len(prep_tegen)-int(train_size_proportion*len(prep_tegen))))
    if IMFs.shape[0]==1:
        #Data preprocessing
        #1) creat generator from ts
        int_generator = gen_prep_init(IMFs[0],window_size)
        #2) Dismantling the generator
        X,y = X_y_from_gen(int_generator)
        #3) Train, validation and test splits
        X_train_full, X_test, y_train_full, y_test = X_splitter(X,y,train_test_prop=train_size_proportion)
        X_train, X_validation, y_train, y_validation = X_splitter(X_train_full,y_train_full,train_test_prop = train_val_proportion)
        #4) Scalling dismantaled version
        #Defining scalar on training set. 
        scaler_train = MinMaxScaler()
        arr = X_train_full.reshape(-1,1)
        scalling_array = np.insert(arr,len(arr) , y_train_full[-1]).reshape(-1,1)
        scaler_train = scaler_train.fit(scalling_array)
        #transforming flattened X_train
        scaled_X_full_train = scaler_train.transform(arr)
        #reshaping to intial X_train shape
        scaled_X_full_train = scaled_X_full_train.reshape(X_train_full.shape)
        # Scalling y_train_full
        scaled_y_full_train = scaler_train.transform(y_train_full.reshape(-1,1)).flatten()
        #Scalling X train and validations
        #transforming flattened X_train
        scaled_X_train = scaler_train.transform(X_train.reshape(-1,1))
        scaled_X_validation = scaler_train.transform(X_validation.reshape(-1,1))

        #reshaping to intial X shape
        scaled_X_train = scaled_X_train.reshape(X_train.shape)
        scaled_X_validation = scaled_X_validation.reshape(X_validation.shape)
        #scaled_X_train = scaler_train.transform(y_train_full.reshape(-1,1)).flatten()
        #scaled_X_validation = scaler_train.transform(y_train_full.reshape(-1,1)).flatten()

        #Scalling y train and validations
        scaled_y_train = scaler_train.transform(y_train.reshape(-1,1)).flatten()
        scaled_y_validation = scaler_train.transform(y_validation.reshape(-1,1)).flatten()
        #Defining Scalar for test set
        scaler_test = MinMaxScaler()
        arr_tst = X_test.reshape(-1,1)
        scalling_array_tst = np.insert(arr_tst,len(arr_tst) , y_test[-1]).reshape(-1,1)
        scaler_test = scaler_train.fit(scalling_array_tst)
        #transforming flattened X_test
        scaled_X_test = scaler_test.transform(arr_tst)
        #reshaping to intial X_test shape
        scaled_X_test = scaled_X_test.reshape(X_test.shape)
        # Scalling y_test
        scaled_y_test = scaler_train.transform(y_test.reshape(-1,1)).flatten()

        #Defining the generators
        #batch_size = 1
        scaled_train_full_generator = gen_prep_2(np.vstack((scaled_X_full_train, np.zeros((scaled_X_full_train.shape[1])))),np.insert(scaled_y_full_train,0,0),batch_size = batch_size,num_features = 1)
        scaled_train_generator = gen_prep_2(np.vstack((scaled_X_train, np.zeros((scaled_X_train.shape[1])))),np.insert(scaled_y_train,0,0),batch_size = batch_size,num_features = 1)
        scaled_validation_generator = gen_prep_2(np.vstack((scaled_X_validation, np.zeros((scaled_X_validation.shape[1])))),np.insert(scaled_y_validation,0,0),batch_size = batch_size,num_features = 1)
        scaled_test_generator = gen_prep_2(np.vstack((scaled_X_test, np.zeros((scaled_X_test.shape[1])))),np.insert(scaled_y_test,0,0),batch_size = batch_size,num_features = 1)
        #Defining function to be tunned
        def build_model_lstm(hp):
            """Lstm function coming together"""
            model = tf.keras.Sequential()  
          # Tune the number of units in the LSTM cell
          # Choose an optimal value between 32-512
            #units1 = hp.Int('units1', min_value=2, max_value=4, step=2)
            #units2 = hp.Int('units2', min_value=1, max_value=5, step =1)
           # hp_recurrent_dropout = hp.Choice('recurrent_dropout',values = [0.0,0.1]) #recurrent_dropout
            #hp_dropout= hp.Choice('dropout',values = [0.0,0.1])dropout = hp_dropout ,recurrent_dropout= hp_recurrent_dropout,
            #hp_activation = hp.Choice('activation',["sigmoid", "relu"])
            model.add(LSTM(units=64, activation= "sigmoid",input_shape = scaled_train_full_generator[0][0].shape[1:]))
            #model.add(LSTM(units = 16, activation = "sigmoid"))
            model.add(Dense(1, kernel_initializer='normal', activation='linear'))
            model.compile(loss="mean_squared_error")
          # Tune the learning rate for the optimizer
          # Choose an optimal value from 0.01, 0.001, or 0.0001
            hp_learning_rate = hp.Choice('learning_rate', values=[1e-4])

            model.compile(
              optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
              loss="mean_squared_error",
              metrics=["mean_squared_error"]
          )

            return model

        ### Defining tuner based on predefined model structure for optimizing hyper parameters#####
        tuner = kt.BayesianOptimization(build_model_lstm,objective="mean_squared_error",max_trials=max_trials,
        directory='keras_tuner_dir', project_name='keras_tuner_d61',overwrite = True)
        tuner.search(scaled_train_full_generator, epochs=tuner_epochs,  validation_data= scaled_validation_generator,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',  
              patience=10),tf.keras.callbacks.TensorBoard('my_dir')])
        #Train predictive model on best hp's with monitored validation
        best_hp = tuner.get_best_hyperparameters()[0]
        model = build_model_lstm(best_hp)
        history = model.fit(scaled_train_generator, validation_data= scaled_validation_generator,epochs=epochs)
        # Training model on full training set for predictions (Model to be saved)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        mc = tf.keras.callbacks.ModelCheckpoint(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/TPE_{model_name}.h5', monitor='loss', mode='min', verbose=1, save_best_only=True)
        # fit model
        #history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=4000, verbose=0, callbacks=[es, mc])
        best_hp = tuner.get_best_hyperparameters()[0]
        model = build_model_lstm(best_hp)
        history_full = model.fit(scaled_X_generator,epochs=epochs,callbacks=[es,mc])
        #model, history = fit_model(train_generator, test_generator, window_size = window_size,epochs=epochs)
        if saving == True:
            #model.save(f"{model_name}.hdf5")
            #model.save(f"/content/gdrive/MyDrive/saved models/{model_name}.hdf5")
            #print(f"{model_name} saved successfully")
            hist_df = pd.DataFrame(history.history)
            #hist_df.to_csv(f"{model_name}.csv")
            hist_df.to_csv(f"/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/hist_{model_name}.csv")
            print(f"{model_name} history saved successfully")
            hist_full_df = pd.DataFrame(history_full.history)
            #hist_df.to_csv(f"{model_name}.csv")
            hist_full_df.to_csv(f"/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/hist_full{model_name}.csv")
            print(f"{model_name} history saved successfully")
        #train_loss,val_loss = model.evaluate(train_generator,verbose=0)[0],model.evaluate(test_generator,verbose=0)[1]
       
        prediction_train = scaler_train.inverse_transform(prediction(model1,scaled_train_full_generator))# making predictions on train set
        prediction_test = scaler_test.inverse_transform(prediction(model1,scaled_test_generator)) # making predictions on test set
        #train_true = series.values[:len(full_train)][len(prediction_train)*-1:] #extracting y_train for the orginal data
        #test_true = series.values[len(full_train):][len(prediction_test)*-1:]   #extracting y_test for the orignal data
        model_history = history
        data_training = np.concatenate((y_train_full.reshape(-1,1),prediction_train), axis = 1)
        train_index = series.iloc[:len(y_train_full)].iloc[prediction_train.shape[0]*-1:].index
        data_frame_training = pd.DataFrame(data =data_training,columns =['actual','predicted'],index = train_index)
        #Testing
        data_test = np.concatenate((y_test.reshape(-1,1),prediction_test), axis = 1)
        test_index = series.iloc[len(y_train_full):].iloc[prediction_test.shape[0]*-1:].index
        data_frame_testing = pd.DataFrame(data =data_test,columns =['actual','predicted'],index = test_index)

    else:
        #saving IMF's for later plotting
        np.save(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/IMFs_{model_name}.npy', IMFs)
        train_prediction_IMFs = np.zeros((int(train_size_proportion*len(prep_tegen)),2,IMFs.shape[0]))
        test_prediction_IMFs = np.zeros((len(prep_tegen)-int(train_size_proportion*len(prep_tegen)),2,IMFs.shape[0]))
        IMFs_history = []
        for i in range(IMFs.shape[0]):  # Looping over each IMF 
            #Data preprocessing
            #1) create generator from ts
            int_generator = gen_prep_init(IMFs[i],window_size=window_size)
            #2) Dismantling the generator
            X,y = X_y_from_gen(int_generator)
            #3) Train, validation and test splits
            X_train_full, X_test, y_train_full, y_test = X_splitter(X,y,train_test_prop=train_size_proportion)
            X_train, X_validation, y_train, y_validation = X_splitter(X_train_full,y_train_full,train_test_prop=train_val_proportion)
            #4) Scalling dismantaled version
            #Defining scalar on training set. 
            scaler_train = MinMaxScaler()
            arr = X_train_full.reshape(-1,1)
            scalling_array = np.insert(arr,len(arr) , y_train_full[-1]).reshape(-1,1)
            scaler_train = scaler_train.fit(scalling_array)
            #transforming flattened X_train
            scaled_X_full_train = scaler_train.transform(arr)
            #reshaping to intial X_train shape
            scaled_X_full_train = scaled_X_full_train.reshape(X_train_full.shape)
            # Scalling y_train_full
            scaled_y_full_train = scaler_train.transform(y_train_full.reshape(-1,1)).flatten()
            #Scalling X train and validations
            #transforming flattened X_train
            scaled_X_train = scaler_train.transform(X_train.reshape(-1,1))
            scaled_X_validation = scaler_train.transform(X_validation.reshape(-1,1))

            #reshaping to intial X shape
            scaled_X_train = scaled_X_train.reshape(X_train.shape)
            scaled_X_validation = scaled_X_validation.reshape(X_validation.shape)
            #scaled_X_train = scaler_train.transform(y_train_full.reshape(-1,1)).flatten()
            #scaled_X_validation = scaler_train.transform(y_train_full.reshape(-1,1)).flatten()

            #Scalling y train and validations
            scaled_y_train = scaler_train.transform(y_train.reshape(-1,1)).flatten()
            scaled_y_validation = scaler_train.transform(y_validation.reshape(-1,1)).flatten()
            #Defining Scalar for test set
            scaler_test = MinMaxScaler()
            arr_tst = X_test.reshape(-1,1)
            scalling_array_tst = np.insert(arr_tst,len(arr_tst) , y_test[-1]).reshape(-1,1)
            scaler_test = scaler_train.fit(scalling_array_tst)
            #transforming flattened X_test
            scaled_X_test = scaler_test.transform(arr_tst)
            #reshaping to intial X_test shape
            scaled_X_test = scaled_X_test.reshape(X_test.shape)
            # Scalling y_test
            scaled_y_test = scaler_train.transform(y_test.reshape(-1,1)).flatten()
            #Defining the generators
            #batch_size = 1
            scaled_train_full_generator = gen_prep_2(np.vstack((scaled_X_full_train, np.zeros((scaled_X_full_train.shape[1])))),np.insert(scaled_y_full_train,0,0),batch_size = batch_size,num_features = 1)
            scaled_train_generator = gen_prep_2(np.vstack((scaled_X_train, np.zeros((scaled_X_train.shape[1])))),np.insert(scaled_y_train,0,0),batch_size = batch_size,num_features = 1)
            scaled_validation_generator = gen_prep_2(np.vstack((scaled_X_validation, np.zeros((scaled_X_validation.shape[1])))),np.insert(scaled_y_validation,0,0),batch_size = batch_size,num_features = 1)
            scaled_test_generator = gen_prep_2(np.vstack((scaled_X_test, np.zeros((scaled_X_test.shape[1])))),np.insert(scaled_y_test,0,0),batch_size = batch_size,num_features = 1)
           #Data preprocessing
           #Defining function to be tunned
            def build_model_lstm(hp):
                """Lstm function coming together"""
                model = tf.keras.Sequential()  
              # Tune the number of units in the LSTM cell
              # Choose an optimal value between 32-512
                #units1 = hp.Int('units1', min_value=2, max_value=4, step=2)
                #units2 = hp.Int('units2', min_value=1, max_value=5, step =1)
                #hp_recurrent_dropout = hp.Choice('recurrent_dropout',values = [0.0,0.1]) #recurrent_dropout
                #hp_dropout= hp.Choice('dropout',values = [0.0,0.1]) ,dropout = hp_dropout, recurrent_dropout =  hp_recurrent_dropout
                #hp_activation = hp.Choice('activation',["sigmoid", "relu"])
                model.add(LSTM(units=64,activation= "sigmoid", input_shape = scaled_train_full_generator[0][0].shape[1:]))
                #model.add(LSTM(units=16,activation= "sigmoid"))
                model.add(Dense(1, kernel_initializer='normal', activation='linear'))
                model.compile(loss="mean_squared_error")
              # Tune the learning rate for the optimizer
              # Choose an optimal value from 0.01, 0.001, or 0.0001
                hp_learning_rate = hp.Choice('learning_rate', values=[1e-4])

                model.compile(
                  optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss="mean_squared_error",
                  metrics=["mean_squared_error"]
              )

                return model


            tuner = kt.BayesianOptimization(build_model_lstm,objective="mean_squared_error", max_trials=max_trials,directory='keras_tuner_dir', project_name='keras_tuner_d61',overwrite = True)
            tuner.search(scaled_train_full_generator, epochs=tuner_epochs,  validation_data= scaled_validation_generator,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',  
              patience=10),tf.keras.callbacks.TensorBoard('my_dir')])
            #Train predictive model on best hp's with monitored validation
            best_hp = tuner.get_best_hyperparameters()[0]
            model = build_model_lstm(best_hp)
            history = model.fit(scaled_train_generator, validation_data= scaled_validation_generator,epochs=epochs)
            # Training model on full training set for predictions (Model to be saved)
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
            mc = tf.keras.callbacks.ModelCheckpoint(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/TPE_{model_name}_IMF{i+1}.h5', monitor='loss', mode='min', verbose=1, save_best_only=True)
            # fit model
            #history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=4000, verbose=0, callbacks=[es, mc])
            best_hp = tuner.get_best_hyperparameters()[0]
            model = build_model_lstm(best_hp)
            history_full = model.fit(scaled_X_generator,epochs=epochs,callbacks=[es,mc])
            #model, history = fit_model(train_generator, test_generator, window_size = window_size,epochs=epochs)
            print(f"{model_name}_IMF{i+1} saved successfully")
            if saving == True:
                #model2.save(f"{model_name}_IMF{i+1}.hdf5")
                #model2.save(f"/content/gdrive/MyDrive/saved models/{model_name}_IMF{i+1}.hdf5")
                hist_df = pd.DataFrame(history.history)
                #hist_df.to_csv(f"{model_name}_IMF{i+1}.csv")
                hist_df.to_csv(f"/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/hist_{model_name}_IMF{i+1}.csv")
                print(f"{model_name}_IMF{i+1}_history saved successfully")
                hist_full_df = pd.DataFrame(history_full.history)
                #hist_df.to_csv(f"{model_name}_IMF{i+1}.csv")
                hist_full_df.to_csv(f"/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/full_hist_{model_name}_IMF{i+1}.csv")
                print(f"full_hist_{model_name}_IMF{i+1}_history saved successfully")
            #train_loss,val_loss = model.evaluate(train_generator,verbose=0)[0],model.evaluate(test_generator,verbose=0)[1]
            IMFs_history.append(history)
            prediction_train = scaler_train.inverse_transform(prediction(model2,scaled_train_full_generator))# making predictions on train set
            prediction_test = scaler_test.inverse_transform(prediction(model2,scaled_test_generator)) # making predictions on test set
            train_prediction_total += prediction_train.flatten()
            #y_train = scaler_train.inverse_transform(y_from_gen(full_train_generator)) # target for the training set        
                #y_test =  scaler_test.inverse_transform(y_from_gen(test_generator)) # target for the test set 
            
            true_pred_train = np.concatenate((y_train_full.reshape(-1,1),prediction_train.reshape(-1,1)),axis=1)
            true_pred_test = np.concatenate((y_test.reshape(-1,1),prediction_test.reshape(-1,1)),axis=1)
            test_prediction_total += prediction_test.flatten()
            train_prediction_IMFs[:,:,i] = true_pred_train
            test_prediction_IMFs[:,:,i] = true_pred_test
      #Rework the indices here to account for window size###    
        #train_true = series.values[window_size:len(y_train)][len(full_train)*-1:] #extracting y_train for the orginal data
        #test_true = series.values[len(full_train):]#[len(test_prediction_total)*-1:]   #extracting y_test for the orignal data

        data_training = np.concatenate((y_train_full.reshape(-1,1),train_prediction_total.reshape(-1,1)), axis = 1)
        data_train = data_training.reshape((data_training.shape[0],data_training.shape[1],1))
        IMFs_training_total_true_pred = np.concatenate((train_prediction_IMFs,data_train),axis=2)
        train_index = series.iloc[:len(y_train_full)].iloc[train_prediction_total.reshape(-1,1).shape[0]*-1:].index
        data_frame_training = pd.DataFrame(data =data_training,columns =['actual','predicted'],index = train_index)
        #Testing
        data_test = np.concatenate((y_test.reshape(-1,1),test_prediction_total.reshape(-1,1)), axis = 1)
        data_tst = data_test.reshape((data_test.shape[0],data_test.shape[1],1))
        IMFs_test_total_true_pred = np.concatenate((test_prediction_IMFs,data_tst),axis=2)
        test_index = series.iloc[len(y_train_full):].iloc[test_prediction_total.reshape(-1,1).shape[0]*-1:].index
        data_frame_testing = pd.DataFrame(data =data_test,columns =['actual','predicted'],index = test_index)
        

    if Decompose==True:
        np.save(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/{model_name}_IMFs_train_predict.npy', IMFs_training_total_true_pred)
        np.save(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/{model_name}_IMFs_test_predict.npy', IMFs_test_total_true_pred)
        np.save(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/{model_name}_Comb_train_predict.npy', data_frame_training)
        np.save(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/{model_name}_Comb_test_predict.npy', data_frame_testing)
        

        
        return (IMFs,IMFs_history,IMFs_training_total_true_pred,IMFs_test_total_true_pred,data_frame_training,data_frame_testing)
    else:
        np.save(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/{model_name}_train_predict.npy', data_frame_training)
        np.save(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/{model_name}_test_predict.npy', data_frame_testing)
        return (model_history,data_frame_training,data_frame_testing)



# Generate 3 evaluation matrix for each true, predicted pair
def IMFs_evaluation_table(IMFs_Tensor):
    combined = pd.DataFrame()
    for i in range(IMFs_Tensor.shape[2]-1):
        combined[f'IMF{i+1}'] = evaluation(IMFs_Tensor[:,0,i],IMFs_Tensor[:,1,i],model_name = f'IMF{i+1}')[f'IMF{i+1}']
    return(combined.T)
#### Functions to facilitate distributions of evaluation (performance) metrics
# IMF BASED
# RMSE
def RMSE_imfs_evaluation_combined(EEMD_LSTM,WD_EEMD_LSTM,CEEMDAN_LSTM,WD_CEEMDAN_LSTM,EWT_LSTM,WD_EWT_LSTM):
    RMSE_EEMD_LSTM = IMFs_evaluation_table(EEMD_LSTM[3])[['RMSE']].rename(columns={'RMSE':'EEMD_LSTM'})
    RMSE_WD_EEMD_LSTM = IMFs_evaluation_table(WD_EEMD_LSTM[3])[['RMSE']].rename(columns={'RMSE':'WD_EEMD_LSTM'})
    RMSE_CEEMDAN_LSTM = IMFs_evaluation_table(CEEMDAN_LSTM[3])[['RMSE']].rename(columns={'RMSE':'CEEMDAN_LSTM'})
    RMSE_WD_CEEMDAN_LSTM = IMFs_evaluation_table(WD_CEEMDAN_LSTM[3])[['RMSE']].rename(columns={'RMSE':'WD_CEEMDAN_LSTM'})
    RMSE_EWT_LSTM = IMFs_evaluation_table(EWT_LSTM[3])[['RMSE']].rename(columns={'RMSE':'EWT_LSTM'})
    RMSE_WD_EWT_LSTM = IMFs_evaluation_table(WD_EWT_LSTM[3])[['RMSE']].rename(columns={'RMSE':'WD_EWT_LSTM'})
    RMSE_imfs_evaluation_combined = RMSE_EEMD_LSTM.copy()
    RMSE_imfs_evaluation_combined['WD_EEMD_LSTM'] = RMSE_WD_EEMD_LSTM['WD_EEMD_LSTM']
    RMSE_imfs_evaluation_combined['CEEMDAN_LSTM'] = RMSE_CEEMDAN_LSTM['CEEMDAN_LSTM']
    RMSE_imfs_evaluation_combined['WD_CEEMDAN_LSTM'] = RMSE_WD_CEEMDAN_LSTM['WD_CEEMDAN_LSTM']
    RMSE_imfs_evaluation_combined['EWT_LSTM'] = RMSE_EWT_LSTM['EWT_LSTM']
    RMSE_imfs_evaluation_combined['WD_EWT_LSTM'] = RMSE_WD_EWT_LSTM['WD_EWT_LSTM']
    return RMSE_imfs_evaluation_combined
#r squared
def r2_imfs_evaluation_combined(EEMD_LSTM,WD_EEMD_LSTM,CEEMDAN_LSTM,WD_CEEMDAN_LSTM,EWT_LSTM,WD_EWT_LSTM):
    r2_EEMD_LSTM = IMFs_evaluation_table(EEMD_LSTM[3])[['r_squared']].rename(columns={'r_squared':'EEMD_LSTM'})
    r2_WD_EEMD_LSTM = IMFs_evaluation_table(WD_EEMD_LSTM[3])[['r_squared']].rename(columns={'r_squared':'WD_EEMD_LSTM'})
    r2_CEEMDAN_LSTM = IMFs_evaluation_table(CEEMDAN_LSTM[3])[['r_squared']].rename(columns={'r_squared':'CEEMDAN_LSTM'})
    r2_WD_CEEMDAN_LSTM = IMFs_evaluation_table(WD_CEEMDAN_LSTM[3])[['r_squared']].rename(columns={'r_squared':'WD_CEEMDAN_LSTM'})
    r2_EWT_LSTM = IMFs_evaluation_table(EWT_LSTM[3])[['r_squared']].rename(columns={'r_squared':'EWT_LSTM'})
    r2_WD_EWT_LSTM = IMFs_evaluation_table(WD_EWT_LSTM[3])[['r_squared']].rename(columns={'r_squared':'WD_EWT_LSTM'})
    r2_imfs_evaluation_combined = r2_EEMD_LSTM.copy()
    r2_imfs_evaluation_combined['WD_EEMD_LSTM'] = r2_WD_EEMD_LSTM['WD_EEMD_LSTM']
    r2_imfs_evaluation_combined['CEEMDAN_LSTM'] = r2_CEEMDAN_LSTM['CEEMDAN_LSTM']
    r2_imfs_evaluation_combined['WD_CEEMDAN_LSTM'] = r2_WD_CEEMDAN_LSTM['WD_CEEMDAN_LSTM']
    r2_imfs_evaluation_combined['EWT_LSTM'] = r2_EWT_LSTM['EWT_LSTM']
    r2_imfs_evaluation_combined['WD_EWT_LSTM'] = r2_WD_EWT_LSTM['WD_EWT_LSTM']
    return r2_imfs_evaluation_combined
#Directional symmetry (DS)
def DS_imfs_evaluation_combined(EEMD_LSTM,WD_EEMD_LSTM,CEEMDAN_LSTM,WD_CEEMDAN_LSTM,EWT_LSTM,WD_EWT_LSTM):
    DS_EEMD_LSTM = IMFs_evaluation_table(EEMD_LSTM[3])[['DS']].rename(columns={'DS':'EEMD_LSTM'})
    DS_WD_EEMD_LSTM = IMFs_evaluation_table(WD_EEMD_LSTM[3])[['DS']].rename(columns={'DS':'WD_EEMD_LSTM'})
    DS_CEEMDAN_LSTM = IMFs_evaluation_table(CEEMDAN_LSTM[3])[['DS']].rename(columns={'DS':'CEEMDAN_LSTM'})
    DS_WD_CEEMDAN_LSTM = IMFs_evaluation_table(WD_CEEMDAN_LSTM[3])[['DS']].rename(columns={'DS':'WD_CEEMDAN_LSTM'})
    DS_EWT_LSTM = IMFs_evaluation_table(EWT_LSTM[3])[['DS']].rename(columns={'DS':'EWT_LSTM'})
    DS_WD_EWT_LSTM = IMFs_evaluation_table(WD_EWT_LSTM[3])[['DS']].rename(columns={'DS':'WD_EWT_LSTM'})
    DS_imfs_evaluation_combined = DS_EEMD_LSTM.copy()
    DS_imfs_evaluation_combined['WD_EEMD_LSTM'] = DS_WD_EEMD_LSTM['WD_EEMD_LSTM']
    DS_imfs_evaluation_combined['CEEMDAN_LSTM'] = DS_CEEMDAN_LSTM['CEEMDAN_LSTM']
    DS_imfs_evaluation_combined['WD_CEEMDAN_LSTM'] = DS_WD_CEEMDAN_LSTM['WD_CEEMDAN_LSTM']
    DS_imfs_evaluation_combined['EWT_LSTM'] = DS_EWT_LSTM['EWT_LSTM']
    DS_imfs_evaluation_combined['WD_EWT_LSTM'] = DS_WD_EWT_LSTM['WD_EWT_LSTM']
    return DS_imfs_evaluation_combined


#All models
def models_true_predict_combined(AD_LSTM,WD_LSTM,EEMD_LSTM,WD_EEMD_LSTM,CEEMDAN_LSTM,WD_CEEMDAN_LSTM,EWT_LSTM,WD_EWT_LSTM,i=''):
#### Evaluation Results Combined
    df_LSTM = AD_LSTM[2].rename(columns={'predicted':f'LSTM_Pred{i}'})
    df_WD_LSTM = WD_LSTM[2].rename(columns={'predicted':'WD_LSTM_Pred'})
    df_EEMD_LSTM = EEMD_LSTM[5].rename(columns={'predicted':'EEMD_LSTM_Pred'})
    df_WD_EEMD_LSTM = WD_EEMD_LSTM[5].rename(columns={'predicted':'WD_EEMD_LSTM_Pred'})
    df_CEEMDAN_LSTM = CEEMDAN_LSTM[5].rename(columns={'predicted':'CEEMDAN_LSTM_Pred'})
    df_WD_CEEMDAN_LSTM = WD_CEEMDAN_LSTM[5].rename(columns={'predicted':'WD_CEEMDAN_LSTM_Pred'})
    df_EWT_LSTM = EWT_LSTM[5].rename(columns={'predicted':'EWT_LSTM_Pred'})
    df_WD_EWT_LSTM = WD_EWT_LSTM[5].rename(columns={'predicted':'WD_EWT_LSTM_Pred'})
    prediction_table_combined = df_LSTM.copy()
    prediction_table_combined[f'WD_LSTM_Pred{i}'] = df_WD_LSTM['WD_LSTM_Pred']
    prediction_table_combined[f'EEMD_LSTM_Pred{i}'] = df_EEMD_LSTM['EEMD_LSTM_Pred']
    prediction_table_combined[f'WD_EEMD_LSTM_Pred{i}'] = df_WD_EEMD_LSTM['WD_EEMD_LSTM_Pred']
    prediction_table_combined[f'CEEMDAN_LSTM_Pred{i}'] = df_CEEMDAN_LSTM['CEEMDAN_LSTM_Pred']
    prediction_table_combined[f'WD_CEEMDAN_LSTM_Pred{i}'] = df_WD_CEEMDAN_LSTM['WD_CEEMDAN_LSTM_Pred']
    prediction_table_combined[f'EWT_LSTM_Pred{i}'] = df_EWT_LSTM['EWT_LSTM_Pred']
    prediction_table_combined[f'WD_EWT_LSTM_Pred{i}'] = df_WD_EWT_LSTM['WD_EWT_LSTM_Pred']
    return prediction_table_combined

def all_models_results(series_df,epochs=2, window_size = 10,train_size_proportion = 0.7, batch_size=5,i=''):
    AD_LSTM = complete_model(series_df,window_size = window_size, train_size_proportion = train_size_proportion, batch_size=batch_size,epochs=epochs,Waveletdenoise = False,Decompose = False)
    # Denoised
    WD_LSTM = complete_model(series_df,window_size = window_size, train_size_proportion = train_size_proportion,batch_size=batch_size,epochs=epochs,Waveletdenoise = True,Decompose = False)
    # Decomposed on actual observations
    EEMD_LSTM = complete_model(series_df,window_size = window_size, train_size_proportion = train_size_proportion,batch_size=batch_size,epochs=epochs,Waveletdenoise = False,Decompose = True,decomp_approach='EEMD')
    # Decomposed on denoised observations
    WD_EEMD_LSTM = complete_model(series_df,window_size = window_size, train_size_proportion = train_size_proportion,batch_size=batch_size,epochs=epochs,Waveletdenoise = True,Decompose = True,decomp_approach='EEMD')
    # Decomposed without denoising.
    CEEMDAN_LSTM = complete_model(series_df,window_size = window_size, train_size_proportion = train_size_proportion,batch_size=batch_size,epochs=epochs,Waveletdenoise = False,Decompose = True,decomp_approach ='CEEMDAN')
    # Denoised then decompose
    WD_CEEMDAN_LSTM = complete_model(series_df,window_size = window_size, train_size_proportion = train_size_proportion,batch_size=batch_size,epochs=epochs,Waveletdenoise = True,Decompose = True,decomp_approach ='CEEMDAN')
    # Decomposition only
    EWT_LSTM = complete_model(series_df,window_size = window_size, train_size_proportion = train_size_proportion, batch_size=batch_size,epochs=epochs,Waveletdenoise = False,Decompose = True,decomp_approach ='EWT')
    #Denoised then decomposed
    WD_EWT_LSTM = complete_model(series_df,window_size = window_size, train_size_proportion = train_size_proportion, batch_size=batch_size,epochs=epochs,Waveletdenoise = True,Decompose = True,decomp_approach ='EWT')
    # IMF based evaluation metrics
    RMSE_imfs_combined = RMSE_imfs_evaluation_combined(EEMD_LSTM,WD_EEMD_LSTM,CEEMDAN_LSTM,WD_CEEMDAN_LSTM,EWT_LSTM,WD_EWT_LSTM)
    DS_imfs_combined = DS_imfs_evaluation_combined(EEMD_LSTM,WD_EEMD_LSTM,CEEMDAN_LSTM,WD_CEEMDAN_LSTM,EWT_LSTM,WD_EWT_LSTM)
    r2_imfs_combined = r2_imfs_evaluation_combined(EEMD_LSTM,WD_EEMD_LSTM,CEEMDAN_LSTM,WD_CEEMDAN_LSTM,EWT_LSTM,WD_EWT_LSTM)
    prediction_table_combined = models_true_predict_combined(AD_LSTM,WD_LSTM,EEMD_LSTM,WD_EEMD_LSTM,CEEMDAN_LSTM,WD_CEEMDAN_LSTM,EWT_LSTM,WD_EWT_LSTM,i=i)
    return RMSE_imfs_combined, DS_imfs_combined, r2_imfs_combined, prediction_table_combined

###
def Metrics_distributions(Time_series_df,epochs = 1, sample_size = 3 ,number_of_models = 8, calculated_metrics = 3,batch_size=15,window_size=7):   
    tic = time.perf_counter()
    dist_evaluations = np.zeros((number_of_models,calculated_metrics,sample_size))
    RMSE_IMFs_dist = pd.DataFrame() #np.zeros((No_IMFs,number_of_models,sample_size))
    DS_IMFs_dist = pd.DataFrame()#np.zeros((No_IMFs,number_of_models,sample_size))
    r2_IMFs_dist = pd.DataFrame()#np.zeros((No_IMFs,number_of_models,sample_size))
    for i in range(sample_size):
        RMSE_IMFs,DS_IMFs,r2_IMFs,actual_pred_all_models = all_models_results(Time_series_df,epochs=epochs,batch_size=batch_size,window_size=window_size)
        full_evaluation = all_models_evaluate(actual_pred_all_models.dropna()).T
        #full_evaluation = full_evaluation.values
        dist_evaluations[:,:,i] = full_evaluation.values
        RMSE_IMFs_dist[[f'EEMD_LSTM{i+1}',f'WD_EEMD_LSTM{i+1}',f'CEEMDAN_LSTM{i+1}',f'WD_CEEMDAN_LSTM{i+1}',f'EWT_LSTM{i+1}',f'WD_EWT_LSTM{i+1}']] = RMSE_IMFs
        DS_IMFs_dist[[f'EEMD_LSTM{i+1}',f'WD_EEMD_LSTM{i+1}',f'CEEMDAN_LSTM{i+1}',f'WD_CEEMDAN_LSTM{i+1}',f'EWT_LSTM{i+1}',f'WD_EWT_LSTM{i+1}']] = DS_IMFs
        r2_IMFs_dist[[f'EEMD_LSTM{i+1}',f'WD_EEMD_LSTM{i+1}',f'CEEMDAN_LSTM{i+1}',f'WD_CEEMDAN_LSTM{i+1}',f'EWT_LSTM{i+1}',f'WD_EWT_LSTM{i+1}']] = r2_IMFs
    #columns = full_evaluation.columns
    index = full_evaluation.index # [i for i in range(sample_size)]
    RMSE_dist = pd.DataFrame(data = dist_evaluations[:,0,:], columns = [f'RMSE{i+1}' for i in range(sample_size)], index=index)
    ds_dist = pd.DataFrame(data = dist_evaluations[:,1,:], columns = [f'DS{i+1}' for i in range(sample_size)], index=index)
    r2_dist = pd.DataFrame(data = dist_evaluations[:,2,:], columns = [f'R_Squared{i+1}' for i in range(sample_size)], index=index)
    toc = time.perf_counter()
    print(f"The models run in {(toc - tic)/60:0.3f} minutes ")
    return (RMSE_dist,ds_dist,r2_dist,RMSE_IMFs_dist,DS_IMFs_dist,r2_IMFs_dist)


# Performance metric results
def Model_evaluate(actual_predicted_dataframe,title = 'EEMD_LSTM'):
    """ This is a function build on top of the evaluation function. Its written to make it easier to obtain plots from function outputs
    
        input: data_frame with the actual and predicted values as its columns
        output: RMSE,DS and r_squared values"""
    Actual = actual_predicted_dataframe['actual'].values
    Predicted = actual_predicted_dataframe['predicted'].values
    return evaluation(Actual,Predicted,model_name= title).T


def all_models_evaluate(actual_pred_all_models):
    """ This is a function build on top of the evaluation function. Its written to make it easier to obtain evaluation
        metrics for all models of interest based on the table with actual values and predicted values from each model
    
        input: data_frame with the actual and predicted values of all models as its columns
        output: Table with RMSE, DS and r_squared values for each model of interest """
    actual = actual_pred_all_models['actual'].values
    dataframe = pd.DataFrame()
    for col in actual_pred_all_models.columns[1:]:
        dataframe[col] = evaluation(actual,actual_pred_all_models[col].values,model_name= col)[col]
    return dataframe
    
def metric_distributions_dataframe(values,it=1):
  """This function renames the column names of the metric evaluation table based on each iterative output
  arguments: Values from the table of computed metric of the various models
  returns: Table of computed metric with columns labeled to incoorparate iteration number
  """
  return pd.DataFrame(data = values, index = full_evaluation.index, columns = [f'RMSE{it}',f'DS{it}',f'r_squared{it}'])
  
def metric_distributions(ts_data,epochs=1, window_size = 5, train_size_proportion = 0.7, batch_size=5,runs = 4):
  """This functions retrains multiple models for an extra understanding of the model performances of multi models.
     It is aimed at training all eight models multiple times and storing their performance metrics and predicted 
     outputs in tables.
     Arguments 
        ts data: Used for train, test and evaluate the performance of multiple trained hybrid LSTM models
        epochs: Number of epochs to run for each model 
        Window_size: How far back to look for the next days prediction
        train_size_proportion: Choice of the proportion to choose for training and testing
        runs: The number of times to run each model and compute perfomance metrics and prediction outputs
      Returns: Two tables, one with the multiple metric results and the other with multiple prediction outputs by each model
     """
  dist_evaluations = []
  prediction_dist = [] 
  for i in range(runs):
      actual_pred_all_models = all_models_results(ts_data,epochs=epochs,window_size = window_size,train_size_proportion = train_size_proportion, batch_size=5,i=i)
      full_evaluation = all_models_evaluate(actual_pred_all_models[-1]).T
      #full_evaluation = full_evaluation.values.reshape((full_evaluation.shape[0],full_evaluation.shape[1],1))
      prediction_dist.append(actual_pred_all_models[-1])
      dist_evaluations.append(metric_distributions_dataframe(full_evaluation.values,i))
  multi_runs = pd.concat(dist_evaluations, axis=1)
  models_pred_dist = pd.concat(prediction_dist, axis=1)
  return (multi_runs , models_pred_dist)
  
  # Running all hybrid models for each considered data proportion. This is useful for comparing model performance under different sample sizes 
# 
def all_models_training_on_multiple_data_proportions(start,step,data_series,epochs = 1, batch_size = 50):
  """The function is for training hybrid time series LSTM models on cummulative proportions of the data. This is to check  model performs 
  with increase in samples of a time series.
  
  args: 
  
  Start: The lowest data proportion to be use should be in the range (0,1)
  Step: By how much you want to increase your data proportions as you train each progressive model (0,1)
  Data_series: The data series on which the models are being trained
  The last two arguments are for configuring the hybrid LSTM to be trained. For more info see complete_model under the Decompose_LSTMs module

  Output:
  A list with all neccessary training results for each progressive model. The first element of the list are the results from training on 
  the lowest data proportion.
  """
  proportion_of_data_used = np.arange(start,1.001,step)
  len_data_series = data_series.shape[0]
  results_for_each_data_proportion = []
  for proportion in proportion_of_data_used:
    df = data_series.head(int(len_data_series*proportion))
    results_for_each_data_proportion.append(all_models_results(df,window_size = 7,epochs=epochs,batch_size=batch_size))
  return proportion_of_data_used, results_for_each_data_proportion
def model_performances_per_data_proportion(all_models_multi_prop):
  # Function for extracting the model results summary table for each data proportion
  proportion_based_performance_summary_tables = [all_models_evaluate(all_models_multi_prop[1][i][3].dropna()).T for i in range(len(all_models_multi_prop[1]))]
  #loop to extract and join the Model summary results for each data proportion
  summary_all_proportions = []
  all_models_props = [round(all_models_multi_prop[0][i]*100,2) for i in range(len(all_models_multi_prop[0]))]
  for prop,summary_df in zip(all_models_props,proportion_based_performance_summary_tables): 
    col1 = f"RMSE ({prop}% data)"
    col2 = f"DS({prop}% data)"
    col3 = f"r_squared({prop}% data)"
    summary_df.columns = [col1,col2,col3]
    summary_all_proportions.append(summary_df)
  #summary_all_proportions
  combined_summary_performance = pd.concat(summary_all_proportions,axis=1) 
  #Reshuffling the columns such that each metric is presented separately
  RMSE_combined = combined_summary_performance.filter(regex='RMSE')
  DS_combined = combined_summary_performance.filter(regex='DS')
  r_squared_combined = combined_summary_performance.filter(regex='r_squared')
  return pd.concat([RMSE_combined,DS_combined,r_squared_combined],axis=1)

########################### Models for the non hind sight experiment #####################################################
# Complete model 
def complete_model_train_only(series,window_size = 1,train_val_proportion = 0.7, batch_size=15, epochs = 10,tuner_epochs = 3,max_trials=2, Waveletdenoise = True, Decompose=True, decomp_approach = 'EEMD',saving = True, model_name="WD_EEMD_LSTM"):
    if (type(series) == np.ndarray) or (type(series) == list):
        series = pd.DataFrame(data = series)
    else:
        series = series
    if Waveletdenoise == True:
        wavelet_denoised_series = wave(series,wavelet = 'db8',padding = 'smooth',dlevel = 1,thresh=3) #denoising
    else:
        wavelet_denoised_series = series  
    if (Decompose == True) and (decomp_approach == 'EEMD'):
        IMFs =  EEMD_decomp(wavelet_denoised_series) # decomposing with EEMD
    elif (Decompose == True) and (decomp_approach == 'CEEMDAN'):
        IMFs =  CEEMDAN_decomp(wavelet_denoised_series)
    elif (Decompose == True) and (decomp_approach == 'EWT'):
        IMFs =  EWT_decomp(wavelet_denoised_series)    
    else:
        IMFs = np.array([series.values.flatten()])
        
    #EEMD_based_IMFs =  EEMD_decomp(wavelet_denoised_series) # decomposing with EEMD
    results = []
    index = []
    prep_tegen = gen_prep_init(series.values,window_size=window_size)
    if IMFs.shape[0]==1:
        #Data preprocessing
        #1) creat generator from ts
        int_generator = gen_prep_init(IMFs[0],window_size)
        #2) Dismantling the generator
        X,y = X_y_from_gen(int_generator)
        #3) Train, validation and test splits
        X_train, X_validation, y_train, y_validation = X_splitter(X,y,train_test_prop=train_val_proportion)
        #4) Scalling dismantaled version
        #Defining scalar on training set. 
        scaler_train = MinMaxScaler()
        arr = X.reshape(-1,1)
        scalling_array = np.insert(arr,len(arr) , y_train[-1]).reshape(-1,1)
        scaler_train = scaler_train.fit(scalling_array)
        #transforming flattened X_train
        scaled_X = scaler_train.transform(arr)
        #reshaping to intial X_train shape
        scaled_X = scaled_X.reshape(X.shape)
        # Scalling y_train_full
        scaled_y = scaler_train.transform(y.reshape(-1,1)).flatten()
        #Scalling X train and validations
        #transforming flattened X_train
        scaled_X_train = scaler_train.transform(X_train.reshape(-1,1))
        scaled_X_validation = scaler_train.transform(X_validation.reshape(-1,1))

        #reshaping to intial X shape
        scaled_X_train = scaled_X_train.reshape(X_train.shape)
        scaled_X_validation = scaled_X_validation.reshape(X_validation.shape)
        
        #Scalling y train and validations
        scaled_y_train = scaler_train.transform(y_train.reshape(-1,1)).flatten()
        scaled_y_validation = scaler_train.transform(y_validation.reshape(-1,1)).flatten()
        #Defining the generators
        #batch_size = 1
        scaled_X_generator = gen_prep_2(np.vstack((scaled_X, np.zeros((scaled_X.shape[1])))),np.insert(scaled_y,0,0),batch_size = batch_size,num_features = 1)
        scaled_train_generator = gen_prep_2(np.vstack((scaled_X_train, np.zeros((scaled_X_train.shape[1])))),np.insert(scaled_y_train,0,0),batch_size = batch_size,num_features = 1)
        scaled_validation_generator = gen_prep_2(np.vstack((scaled_X_validation, np.zeros((scaled_X_validation.shape[1])))),np.insert(scaled_y_validation,0,0),batch_size = batch_size,num_features = 1)
        #Defining function to be tunned
        def build_model_lstm(hp):
            """Lstm function coming together"""
            model = tf.keras.Sequential()  
          # Tune the number of units in the LSTM cell
          # Choose an optimal value between 32-512
            units1 = hp.Int('units1', min_value=224, max_value=512, step=32)
            #units2 = hp.Int('units2', min_value=1, max_value=5, step =1)
            #hp_recurrent_dropout = hp.Choice('recurrent_dropout',values = [0.0,0.1]) #recurrent_dropout
            #hp_dropout= hp.Choice('dropout',values = [0.0,0.1])
            hp_activation = hp.Choice('activation',["sigmoid", "relu"])
            model.add(LSTM(units=units1, activation= hp_activation,input_shape = scaled_X_generator[0][0].shape[1:]))
            #model.add(LSTM(units=16,activation= "sigmoid"))
            model.add(Dense(1, kernel_initializer='normal', activation='linear'))
            model.compile(loss="mean_squared_error")
          # Tune the learning rate for the optimizer
          # Choose an optimal value from 0.01, 0.001, or 0.0001
            hp_learning_rate = hp.Choice('learning_rate', values=[1e-4])

            model.compile(
              optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
              loss="mean_squared_error",
              metrics=["mean_squared_error"]
          )

            return model

        ### Defining tuner based on predefined model structure for optimizing hyper parameters##e-###
        tuner = kt.BayesianOptimization(build_model_lstm,objective="mean_squared_error",max_trials=max_trials,
        directory='keras_tuner_dir', project_name='keras_tuner_d61',overwrite = True)
        tuner.search(scaled_train_generator, epochs=tuner_epochs,  validation_data= scaled_validation_generator,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',  
              patience=10),tf.keras.callbacks.TensorBoard('my_dir')])
        #Train predictive model on best hp's with monitored validation
        best_hp = tuner.get_best_hyperparameters()[0]
        model = build_model_lstm(best_hp)
        history = model.fit(scaled_train_generator, validation_data= scaled_validation_generator,epochs=epochs)
        # Training model on full training set for predictions (Model to be saved)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        mc = tf.keras.callbacks.ModelCheckpoint(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Saved_models/TPE_{model_name}.hdf5', monitor='loss', mode='min', verbose=1, save_best_only=True)
        # fit model
        #history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=4000, verbose=0, callbacks=[es, mc])
        best_hp = tuner.get_best_hyperparameters()[0]
        best_hps = np.array([best_hp.get('activation'),best_hp.get('units1'),best_hp.get('learning_rate')])
        df_best_hps = pd.DataFrame(data = best_hps.reshape(1,3),columns=['activation','units1','learning_rate'] )
        #print(best_hps.get('activation'))
        df_best_hps.to_csv(f"/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/BHP_TPE_{model_name}.csv")
        #print(best_hps.get('units1'))
        #best_hps.get('learning_rate')
        model = build_model_lstm(best_hp)
        history_full = model.fit(scaled_X_generator,epochs=epochs,callbacks=[es,mc])
        #model, history = fit_model(train_generator, test_generator, window_size = window_size,epochs=epochs)
        if saving == True:
            #model.save(f"{model_name}.hdf5")
            #model1.save(f"/content/gdrive/MyDrive/saved models/TPE_{model_name}.hdf5")
            #print(f"{model_name} saved successfully")
            hist_df = pd.DataFrame(history.history)
            #hist_df.to_csv(f"{model_name}.csv")
            hist_df.to_csv(f"/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/hist_TPE_{model_name}.csv")
            print(f"hist_{model_name} history saved successfully")
            hist_full_df = pd.DataFrame(history_full.history)
            #hist_df.to_csv(f"{model_name}.csv")
            hist_full_df.to_csv(f"/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/hist_full_TPE_{model_name}.csv")
            print(f"hist_full_{model_name} history saved successfully")
        
    else:
        #saving IMF's for later plotting
        np.save(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/TPE_IMFs_{model_name}.npy', IMFs)
        #train_prediction_IMFs = np.zeros((int(train_size_proportion*len(prep_tegen)),2,IMFs.shape[0]))
        #test_prediction_IMFs = np.zeros((len(prep_tegen)-int(train_size_proportion*len(prep_tegen)),2,IMFs.shape[0]))
        IMFs_best_hp = []#np.zeros((IMFs.shape[0],3))
        index = [f'{decomp_approach}_IMF{i+1}' for i in range(IMFs.shape[0])]
        for i in range(IMFs.shape[0]):  # Looping over each IMF 
            #Data preprocessing
            #1) create generator from ts
            int_generator = gen_prep_init(IMFs[i],window_size=window_size)
            #2) Dismantling the generator
            X,y = X_y_from_gen(int_generator)
            #3) Train, validation and test splits
            X_train, X_validation, y_train, y_validation = X_splitter(X,y,train_test_prop=train_val_proportion)
            #4) Scalling dismantaled version
            #Defining scalar on training set. 
            scaler_train = MinMaxScaler()
            arr = X.reshape(-1,1)
            scalling_array = np.insert(arr,len(arr) , y[-1]).reshape(-1,1)
            scaler_train = scaler_train.fit(scalling_array)
            #transforming flattened X_train
            scaled_X = scaler_train.transform(arr)
            #reshaping to intial X_train shape
            scaled_X = scaled_X.reshape(X.shape)
            # Scalling y_train_full
            scaled_y = scaler_train.transform(y.reshape(-1,1)).flatten()
            #Scalling X train and validations
            #transforming flattened X_train
            scaled_X_train = scaler_train.transform(X_train.reshape(-1,1))
            scaled_X_validation = scaler_train.transform(X_validation.reshape(-1,1))

            #reshaping to intial X shape
            scaled_X_train = scaled_X_train.reshape(X_train.shape)
            scaled_X_validation = scaled_X_validation.reshape(X_validation.shape)
            #scaled_y_train = scaler_train.transform(y_train.reshape(-1,1)).flatten()
            #scaled_X_validation = scaler_train.transform(y_train_full.reshape(-1,1)).flatten()

            #Scalling y train and validations
            scaled_y_train = scaler_train.transform(y_train.reshape(-1,1)).flatten()
            scaled_y_validation = scaler_train.transform(y_validation.reshape(-1,1)).flatten()
            #Defining the generators
            #batch_size = 1
            scaled_X_generator = gen_prep_2(np.vstack((scaled_X, np.zeros((scaled_X.shape[1])))),np.insert(scaled_y,0,0),batch_size = batch_size,num_features = 1)
            scaled_train_generator = gen_prep_2(np.vstack((scaled_X_train, np.zeros((scaled_X_train.shape[1])))),np.insert(scaled_y_train,0,0),batch_size = batch_size,num_features = 1)
            scaled_validation_generator = gen_prep_2(np.vstack((scaled_X_validation, np.zeros((scaled_X_validation.shape[1])))),np.insert(scaled_y_validation,0,0),batch_size = batch_size,num_features = 1)
            #scaled_test_generator = gen_prep_2(np.vstack((scaled_X_test, np.zeros((scaled_X_test.shape[1])))),np.insert(scaled_y_test,0,0),batch_size = batch_size,num_features = 1)
           #Data preprocessing
           #Defining function to be tunned
            def build_model_lstm(hp):
                """Lstm function coming together"""
                model = tf.keras.Sequential()  
              # Tune the number of units in the LSTM cell
              # Choose an optimal value between 32-512
                units1 = hp.Int('units1', min_value=224, max_value=512, step=32)
                #units2 = hp.Int('units2', min_value=1, max_value=5, step =1)
                #hp_recurrent_dropout = hp.Choice('recurrent_dropout',values = [0.0, 0.1]) #recurrent_dropout
                #hp_dropout= hp.Choice('dropout',values = [0.0,0.1])
                hp_activation = hp.Choice('activation',["sigmoid", "relu"])
                model.add(LSTM(units=units1, activation= hp_activation,input_shape = scaled_X_generator[0][0].shape[1:]))
                #model.add(LSTM(units = 16,activation = "sigmoid",)) #,dropout = hp_dropout, recurrent_dropout = hp_recurrent_dropout
                model.add(Dense(1, kernel_initializer='normal', activation='linear'))
                model.compile(loss="mean_squared_error")
              # Tune the learning rate for the optimizer
              # Choose an optimal value from 0.01, 0.001, or 0.0001
                hp_learning_rate = hp.Choice('learning_rate', values=[1e-4])

                model.compile(
                  optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss="mean_squared_error",
                  metrics=["mean_squared_error"]
              )

                return model


            tuner = kt.BayesianOptimization(build_model_lstm,objective="mean_squared_error", max_trials=max_trials,directory='keras_tuner_dir', project_name='keras_tuner_d61',overwrite = True)
            tuner.search(scaled_train_generator, epochs=tuner_epochs,  validation_data= scaled_validation_generator,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',  
              patience=10),tf.keras.callbacks.TensorBoard('my_dir')])
            #Train predictive model on best hp's with monitored validation
            best_hp = tuner.get_best_hyperparameters()[0]
            IMFs_best_hp.append([best_hp.get('activation'),best_hp.get('units1'),best_hp.get('learning_rate')])
            model = build_model_lstm(best_hp)
            history = model.fit(scaled_train_generator, validation_data= scaled_validation_generator,epochs=epochs)
            # Training model on full training set for predictions (Model to be saved)
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
            mc = tf.keras.callbacks.ModelCheckpoint(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Saved_models/TPE_{model_name}_IMF{i+1}.hdf5', monitor='loss', mode='min', verbose=1, save_best_only=True)
            # fit model
            #history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=4000, verbose=0, callbacks=[es, mc])
            best_hp = tuner.get_best_hyperparameters()[0]
            model = build_model_lstm(best_hp)
            history_full = model.fit(scaled_X_generator,epochs=epochs,callbacks=[es,mc])
            #model, history = fit_model(train_generator, test_generator, window_size = window_size,epochs=epochs)
            if saving == True:
                #model2.save(f"{model_name}_IMF{i+1}.hdf5")
                #model2.save(f"/content/gdrive/MyDrive/saved models/TPE_{model_name}_IMF{i+1}.hdf5")
                #print(f"{model_name}_IMF{i+1} saved successfully")
                hist_df = pd.DataFrame(history.history)
                #hist_df.to_csv(f"{model_name}_IMF{i+1}.csv")
                hist_df.to_csv(f"/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/hist_TPE_{model_name}_IMF{i+1}.csv")
                print(f"hist_{model_name}_IMF{i+1}_history saved successfully")
                hist_full_df = pd.DataFrame(history_full.history)
                #hist_df.to_csv(f"{model_name}_IMF{i+1}.csv")
                hist_full_df.to_csv(f"/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/hist_full_TPE_{model_name}_IMF{i+1}.csv")
                print(f"hist_full_{model_name}_IMF{i+1}_history saved successfully")
            #train_loss,val_loss = model.evaluate(train_generator,verbose=0)[0],model.evaluate(test_generator,verbose=0)[1]
            #IMFs_history.append(history)
        best_hp_IMF = pd.DataFrame(data = IMFs_best_hp, columns=['activation','units1','learning_rate'],index=index)    
        best_hp_IMF.to_csv(f"/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/{model_name}_Best_HP_IMF.csv")
    if Decompose==True:
        return #"Training and saving successful"
    else:
        return #"Training and saving successful"
    

    ################################################################################################################################################################################################################# Imported IMF's Trainer ###############################################################################
#saving IMF's for later plotting
# Importing IMF's
def imported_imfs_trainer(imfs_table_name,window_size = 1,train_val_proportion = 0.7, batch_size=15, epochs = 10,tuner_epochs = 3,max_trials=2,saving = True, model_name="EEMD_LSTM"):
    df_IMFs = pd.read_csv(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/IMFs/{imfs_table_name}.csv',index_col=0)
    #np.save(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/TPE_IMFs_{model_name}.npy', IMFs)
    IMFs = df_IMFs.values
    IMFs_best_hp = []#np.zeros((IMFs.shape[0],3))
    index = [f'{imfs_table_name}_IMF{i+1}' for i in range(IMFs.shape[0])]
    for i in range(IMFs.shape[0]):  # Looping over each IMF 
        #Data preprocessing
        #1) create generator from ts
        int_generator = gen_prep_init(IMFs[i],window_size=window_size)
        #2) Dismantling the generator
        X,y = X_y_from_gen(int_generator)
        #3) Train, validation and test splits
        X_train, X_validation, y_train, y_validation = X_splitter(X,y,train_test_prop=train_val_proportion)
        #4) Scalling dismantaled version
        #Defining scalar on training set. 
        scaler_train = MinMaxScaler()
        arr = X.reshape(-1,1)
        scalling_array = np.insert(arr,len(arr) , y[-1]).reshape(-1,1)
        scaler_train = scaler_train.fit(scalling_array)
        #transforming flattened X_train
        scaled_X = scaler_train.transform(arr)
        #reshaping to intial X_train shape
        scaled_X = scaled_X.reshape(X.shape)
        # Scalling y_train_full
        scaled_y = scaler_train.transform(y.reshape(-1,1)).flatten()
        #Scalling X train and validations
        #transforming flattened X_train
        scaled_X_train = scaler_train.transform(X_train.reshape(-1,1))
        scaled_X_validation = scaler_train.transform(X_validation.reshape(-1,1))

        #reshaping to intial X shape
        scaled_X_train = scaled_X_train.reshape(X_train.shape)
        scaled_X_validation = scaled_X_validation.reshape(X_validation.shape)
        #scaled_y_train = scaler_train.transform(y_train.reshape(-1,1)).flatten()
        #scaled_X_validation = scaler_train.transform(y_train_full.reshape(-1,1)).flatten()

        #Scalling y train and validations
        scaled_y_train = scaler_train.transform(y_train.reshape(-1,1)).flatten()
        scaled_y_validation = scaler_train.transform(y_validation.reshape(-1,1)).flatten()
        #Defining the generators
        #batch_size = 1
        scaled_X_generator = gen_prep_2(np.vstack((scaled_X, np.zeros((scaled_X.shape[1])))),np.insert(scaled_y,0,0),batch_size = batch_size,num_features = 1)
        scaled_train_generator = gen_prep_2(np.vstack((scaled_X_train, np.zeros((scaled_X_train.shape[1])))),np.insert(scaled_y_train,0,0),batch_size = batch_size,num_features = 1)
        scaled_validation_generator = gen_prep_2(np.vstack((scaled_X_validation, np.zeros((scaled_X_validation.shape[1])))),np.insert(scaled_y_validation,0,0),batch_size = batch_size,num_features = 1)
        #scaled_test_generator = gen_prep_2(np.vstack((scaled_X_test, np.zeros((scaled_X_test.shape[1])))),np.insert(scaled_y_test,0,0),batch_size = batch_size,num_features = 1)
       #Data preprocessing
       #Defining function to be tunned
        def build_model_lstm(hp):
            """Lstm function coming together"""
            model = tf.keras.Sequential()  
          # Tune the number of units in the LSTM cell
          # Choose an optimal value between 32-512
            units1 = hp.Int('units1', min_value=224, max_value=512, step=32)
            #units2 = hp.Int('units2', min_value=1, max_value=5, step =1)
            #hp_recurrent_dropout = hp.Choice('recurrent_dropout',values = [0.0, 0.1]) #recurrent_dropout
            #hp_dropout= hp.Choice('dropout',values = [0.0,0.1])
            hp_activation = hp.Choice('activation',["sigmoid", "relu"])
            model.add(LSTM(units=units1, activation= hp_activation,input_shape = scaled_X_generator[0][0].shape[1:]))
            #model.add(LSTM(units = 16,activation = "sigmoid",)) #,dropout = hp_dropout, recurrent_dropout = hp_recurrent_dropout
            model.add(Dense(1, kernel_initializer='normal', activation='linear'))
            model.compile(loss="mean_squared_error")
          # Tune the learning rate for the optimizer
          # Choose an optimal value from 0.01, 0.001, or 0.0001
            hp_learning_rate = hp.Choice('learning_rate', values=[1e-4])

            model.compile(
              optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
              loss="mean_squared_error",
              metrics=["mean_squared_error"]
          )

            return model


        tuner = kt.BayesianOptimization(build_model_lstm,objective="mean_squared_error", max_trials=max_trials,directory='keras_tuner_dir', project_name='keras_tuner_d61',overwrite = True)
        tuner.search(scaled_train_generator, epochs=tuner_epochs,  validation_data= scaled_validation_generator,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',  
          patience=10),tf.keras.callbacks.TensorBoard('my_dir')])
        #Train predictive model on best hp's with monitored validation
        best_hp = tuner.get_best_hyperparameters()[0]
        IMFs_best_hp.append([best_hp.get('activation'),best_hp.get('units1'),best_hp.get('learning_rate')])
        model = build_model_lstm(best_hp)
        history = model.fit(scaled_train_generator, validation_data= scaled_validation_generator,epochs=epochs)
        # Training model on full training set for predictions (Model to be saved)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        mc = tf.keras.callbacks.ModelCheckpoint(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/IMFs_models/TPE_{model_name}_IMF{i+1}.hdf5', monitor='loss', mode='min', verbose=1, save_best_only=True)
        # fit model
        #history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=4000, verbose=0, callbacks=[es, mc])
        best_hp = tuner.get_best_hyperparameters()[0]
        model = build_model_lstm(best_hp)
        history_full = model.fit(scaled_X_generator,epochs=epochs,callbacks=[es,mc])
        #model, history = fit_model(train_generator, test_generator, window_size = window_size,epochs=epochs)
        if saving == True:
            #model2.save(f"{model_name}_IMF{i+1}.hdf5")
            #model2.save(f"/content/gdrive/MyDrive/saved models/TPE_{model_name}_IMF{i+1}.hdf5")
            #print(f"{model_name}_IMF{i+1} saved successfully")
            hist_df = pd.DataFrame(history.history)
            #hist_df.to_csv(f"{model_name}_IMF{i+1}.csv")
            hist_df.to_csv(f"/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/hist_TPE_{model_name}_IMF{i+1}.csv")
            print(f"hist_{model_name}_IMF{i+1}_history saved successfully")
            hist_full_df = pd.DataFrame(history_full.history)
            #hist_df.to_csv(f"{model_name}_IMF{i+1}.csv")
            hist_full_df.to_csv(f"/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/hist_full_TPE_{model_name}_IMF{i+1}.csv")
            print(f"hist_full_{model_name}_IMF{i+1}_history saved successfully")
        #train_loss,val_loss = model.evaluate(train_generator,verbose=0)[0],model.evaluate(test_generator,verbose=0)[1]
        #IMFs_history.append(history)
    best_hp_IMF = pd.DataFrame(data = IMFs_best_hp, columns=['activation','units1','learning_rate'],index=index)    
    best_hp_IMF.to_csv(f"/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/{model_name}_Best_HP_IMF.csv")
    return #"Training and saving successful"
    
    
################################################################################################################################################################################################################# Prediction functions based on the trained models and the testset########################################

   ################# Support functions for forecasting over test set etc ###################
########################## standard LSTM model support functions############################

def LSTM_single_step_forecast(model,train,window_size):  #predictor_TPE_ND
  predicted = []
  """ Function for obtaining one step forecasting based on the training set when no decomposition is done

  """
  #for i in range(IMFs.shape[0]):
  Test_scaler1 = MinMaxScaler()
  #Fitting scalers on each IMF
  Test_scaler1 = Test_scaler1.fit(train.values.reshape(-1,1))
  #Transforming
  scaled_X_imf1 = Test_scaler1.transform(train.values[-(window_size):].reshape(-1,1))
  #reshaping arrays
  scaled_X_imf1 = scaled_X_imf1.reshape(1,1,window_size)
  #obtaining predictions
  pred1 = model.predict(scaled_X_imf1)
  #inverse_transforms of predictions
  pred1 = Test_scaler1.inverse_transform(pred1.reshape(-1,1))
  #predicted.append(pred1[0])
  return pred1[0][0]
 
def IMFs_onestep_forecast(model_list,IMFs,window_size):
    predicted = []
    """Function for single step forcast when model was trained based on decomposition preprocessing"""
    for i in range(IMFs.shape[0]):
        Test_scaler1 = MinMaxScaler()
        #Fitting scalers on each IMF
        Test_scaler1 = Test_scaler1.fit(IMFs[i,:].reshape(-1,1))
        #Transforming
        scaled_X_imf1 = Test_scaler1.transform(IMFs[i,:][-(window_size):].reshape(-1,1))
        #reshaping arrays
        scaled_X_imf1 = scaled_X_imf1.reshape(1,1,window_size) #change shape to (none,1,2)
        #obtaining predictions
        pred1 = model_list[i].predict(scaled_X_imf1)
        #inverse_transforms of predictions
        pred1 = Test_scaler1.inverse_transform(pred1.reshape(-1,1))
        predicted.append(pred1)
    return np.sum(np.array(predicted))
    
#### Function bringing it all together ##### Train saved forecast increase by 1 training set, train save forecast ...############

def LSTM_forecast_over_test(train, test,window_size = 1,train_val_proportion = 0.7, batch_size=15,epochs = 10,tuner_epochs = 3,max_trials=2, Waveletdenoise = False,SG = False,Decompose=True,decomp_approach='',saving = True, model_name="WD_EEMD_LSTM"):
  """This function facilitates the training and predicting of time series with single LSTM or others 
  based on decomposition preprocessing"""
  predicted = []
  for i in range(test.shape[0]):
      if SG == True:
        trn = train.append(test.head(i+1))
        trn = savgol_filter(trn, window_length=21, polyorder=5, mode="nearest")
        trn=pd.DataFrame(trn,columns=['sg_spi6'])
        trn = train.append(test.head(i+1))
      else:
        trn = train.append(test.head(i+1))
      #print(len(trn))
      #print(len(trn.iloc[:-1]))
      complete_model_train_only(trn.iloc[:-1], window_size = window_size, batch_size = batch_size, train_val_proportion = train_val_proportion, epochs = epochs,tuner_epochs = tuner_epochs, max_trials=max_trials, Waveletdenoise = Waveletdenoise, Decompose = Decompose, decomp_approach = decomp_approach, saving = True, model_name = model_name)
      #import just saved models for next day prediction
      if Decompose == True:
        IMFs = np.load(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/TPE_IMFs_{model_name}.npy')
        model_list = [load_model(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Saved_models/TPE_{model_name}_IMF{i+1}.hdf5') for i in range(IMFs.shape[0])]
        # Performing Predictions from each one then performing single day prediction.
        predicted.append(IMFs_onestep_forecast(model_list,IMFs,window_size=window_size))
      else:
        Model = load_model(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Saved_models/TPE_{model_name}.hdf5')
      # Performing Predictions from each one then performing single day prediction.
        predicted.append(LSTM_single_step_forecast(Model,trn,window_size=window_size))
  return predicted
  ################################################################################################################################################################################################################## imported forecast over imported imfs ##########################################################
  
def LSTM_forecast_over_test_IMFs(test_size,window_size = 1,train_val_proportion = 0.7, batch_size=15,epochs = 10,tuner_epochs = 3,max_trials=2,saving = True, model_name="WD_EEMD_LSTM",Decomposition='EEMD'):
  """This function facilitates the training and predicting of time series with imported IMFs"""
  predicted = []
  for i in range(test_size):
      #trn = train.append(test.head(i+1))
      imported_imfs_trainer(f'{Decomposition}_IMFs{i+1}',window_size = window_size,train_val_proportion = train_val_proportion, batch_size=batch_size, epochs = epochs,tuner_epochs = tuner_epochs,max_trials=max_trials,saving = True, model_name=model_name)
      #print(len(trn))
      IMFs_df = pd.read_csv(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/IMFs/{Decomposition}_IMFs{i+1}.csv',index_col=0)
      IMFs =  IMFs_df.values #np.load(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/TPE_IMFs_{model_name}.npy')
      model_list = [load_model(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/IMFs_models/TPE_{model_name}_IMF{i+1}.hdf5') for i in range(IMFs.shape[0])]
      # Performing Predictions from each one then performing single day prediction.
      predicted.append(IMFs_onestep_forecast(model_list,IMFs,window_size=window_size))
      #Model = load_model(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/IMFs_models/TPE_{model_name}_IMF{i+1}.hdf5')
      # Performing Predictions from each one then performing single day prediction.
      #predicted.append(LSTM_single_step_forecast(Model,trn,window_size=window_size))
  return predicted
#####################################################################################################################
####################################################################################################################

    
def model_training_on_multiple_data_proportions(start,step,data_series,window_size = 1,train_size_proportion = 0.7,train_val_proportion = 0.7, batch_size=15,epochs = 10,tuner_epochs = 3, max_trials=2, Waveletdenoise = True,Decompose=True,decomp_approach = 'EEMD',saving = True, model_name="WD_EEMD_LSTM"):
    """The function is for training hybrid time series LSTM models on cummulative proportions of the data. This is to check  model performs 
  with increase in samples of a time series.
  
    args:  
  
    Start: The lowest data proportion to be use should be in the range (0,1)
    Step: By how much you want to increase your data proportions as you train each progressive model (0,1)
    Data_series: The data series on which the models are being trained
    The last two arguments are for configuring the hybrid LSTM to be trained. For more info see complete_model under the        Decompose_LSTMs module

    Output:
    A list with all neccessary training results for each progressive model. The first element of the list are the results       from training on 
    the lowest data proportion.
  """
    proportion_of_data_used = np.arange(start,1.001,step)
    len_data_series = data_series.shape[0]
    #results_for_each_data_proportion = []
    for proportion in proportion_of_data_used:
        proportion = round(proportion,2)
        df = data_series.head(int(len_data_series*proportion))
        #Test Train split
        train, test = train_test(df,train_size_proportion=train_size_proportion)
        complete_model_train_only(train,window_size = window_size,train_val_proportion = train_val_proportion, batch_size=batch_size,epochs = epochs,tuner_epochs = tuner_epochs,max_trials=max_trials, Waveletdenoise = Waveletdenoise,Decompose=Decompose,decomp_approach = decomp_approach,saving = saving, model_name=f"{proportion}_{model_name}")
        train.to_csv(f"/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/{model_name}_train_at_{proportion}.csv")
        test.to_csv(f"/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/{model_name}_test_at_{proportion}.csv")
        print(f"completed_{proportion*100}%_run")
        #results_for_each_data_proportion.append(all_models_results(df,window_size = 7,epochs=epochs,batch_size=batch_size))
    return "Model run complete and models, dataframes and arrays saved accordingly"
    
    ################# Support functions for prediction etc ###################
def prediction_single_model(model,train,test,window_size =21):
    
    """
    (model) -> Trained time series predictive model " 
    train   -> Timeseries dataframe with training portion
    test    -> Timeseries dataframe with test portion 
    Return array of predicted values based on the training set
    Example:
    >>> prediction_single_model(model,train,test,window_size=3)
        array([y_hat_1,...,y_hat_m])""" 
    #define_scaler
    Test_scaler = MinMaxScaler()
    Test_scaler = Test_scaler.fit(test.values.reshape(-1,1))
    #scaled_init = Test_scaler.transform(train.values[-window_size:].reshape(-1,1))
    predicted = []
    #X = np.insert(train.values,-1,test.values[0])
    train_running = train.copy().values
    i=0
    while i < len(test):
        #from keras.models import load_model
        #model = load_model('model.h5')
        train_run = np.append(train_running,test.values[i])
        X = train_running[-(window_size+1):-1]
        scaled_x = Test_scaler.transform(X.reshape(-1,1))
        scaled_x = scaled_x.reshape(1,1,window_size)
        predicted.append(model.predict(scaled_x))
        i+=1
        train_running = train_run
    predicted = Test_scaler.inverse_transform(np.array(predicted).reshape(-1,1))
    return np.array(predicted)
        
    #######################IMF_predicitions############################
def prediction_5IMF_models(model_list,train,test,window_size =21):
    
    """
    (model) -> Trained time series predictive model " 
    train   -> Timeseries dataframe with training portion
    test    -> Timeseries dataframe with test portion t
    Return array of predicted values based on the training set
    Example:
    >>> prediction_single_model(model,train,test,window_size=3)
        array([y_hat_1,...,y_hat_m])""" 
    #define_scaler
    predicted1 = []
    predicted2 = []
    predicted3 = []
    predicted4 = []
    predicted5 = []
    i=0
    while i < len(test):
        train_running = np.append(train,test.values[i])
        #Obtain IMFs
        IMFs = EEMD_decomp_on_array(train_running)
        #Declaring scalers for each IMF
        Test_scaler1 = MinMaxScaler()
        Test_scaler2 = MinMaxScaler()
        Test_scaler3 = MinMaxScaler()
        Test_scaler4 = MinMaxScaler()
        Test_scaler5 = MinMaxScaler()
        #Fitting scalers on each IMF
        Test_scaler1 = Test_scaler1.fit(IMFs[0,:].reshape(-1,1))
        Test_scaler2 = Test_scaler2.fit(IMFs[1,:].reshape(-1,1))
        Test_scaler3 = Test_scaler3.fit(IMFs[2,:].reshape(-1,1))
        Test_scaler4 = Test_scaler4.fit(IMFs[3,:].reshape(-1,1))
        Test_scaler5 = Test_scaler5.fit(IMFs[4,:].reshape(-1,1))
        #Transforming
        scaled_X_imf1 = Test_scaler1.transform(IMFs[0,:][-(window_size+1):-1].reshape(-1,1))
        scaled_X_imf2 = Test_scaler2.transform(IMFs[1,:][-(window_size+1):-1].reshape(-1,1))
        scaled_X_imf3 = Test_scaler3.transform(IMFs[2,:][-(window_size+1):-1].reshape(-1,1))
        scaled_X_imf4 = Test_scaler4.transform(IMFs[3,:][-(window_size+1):-1].reshape(-1,1))
        scaled_X_imf5 = Test_scaler5.transform(IMFs[4,:][-(window_size+1):-1].reshape(-1,1))
        #reshaping arrays
        scaled_X_imf1 = scaled_X_imf1.reshape(1,1,window_size)
        scaled_X_imf2 = scaled_X_imf2.reshape(1,1,window_size)
        scaled_X_imf3 = scaled_X_imf3.reshape(1,1,window_size)
        scaled_X_imf4 = scaled_X_imf4.reshape(1,1,window_size)
        scaled_X_imf5 = scaled_X_imf5.reshape(1,1,window_size)
        #obtaining predictions
        pred1 = model_list[0].predict(scaled_X_imf1)
        pred2 = model_list[1].predict(scaled_X_imf2)
        pred3 = model_list[2].predict(scaled_X_imf3)
        pred4 = model_list[3].predict(scaled_X_imf4)
        pred5 = model_list[4].predict(scaled_X_imf5)
        #inverse_transforms of predictions
        pred1 = Test_scaler1.inverse_transform(pred1.reshape(-1,1))
        pred2 = Test_scaler2.inverse_transform(pred2.reshape(-1,1))
        pred3 = Test_scaler3.inverse_transform(pred3.reshape(-1,1))
        pred4 = Test_scaler4.inverse_transform(pred4.reshape(-1,1))
        pred5 = Test_scaler5.inverse_transform(pred5.reshape(-1,1))
        #Appending to the predicted lists
        predicted1.append(pred1)
        predicted2.append(pred2)
        predicted3.append(pred3)
        predicted4.append(pred4)
        predicted5.append(pred5)
        #X = train_running[-(window_size+1):-1]
        #scaled_x = Test_scaler.transform(X.reshape(-1,1))
        # Updating training value to new train
        train = train_running
        i+=1
    predicted1 = np.array(predicted1).flatten()
    predicted2 = np.array(predicted2).flatten()
    predicted3 = np.array(predicted3).flatten() 
    predicted4 = np.array(predicted4).flatten()
    predicted5 = np.array(predicted5).flatten() 
    stacked = np.vstack((predicted1,predicted2,predicted3,predicted4,predicted5))
    return (stacked,stacked.sum(axis=0))
    
#Cleaning plotting function
# Function for beautified train loss plots
def train_val_loss_plotter(history_df,Title="Test vs Prediction (Model_name)"):
    fig, ax = plt.subplots(figsize = (14,6))
    # Edit the major and minor ticks of the x and y
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
    # plotting daily data
    # see: https://e2eml.school/matplotlib_lines.html for line colours
    ax.plot(history_df.index, history_df["loss"], color='black', label="Train Loss")
    # plotting 4-day rolling data
    ax.plot(history_df.index, history_df["val_loss"], color='red', linestyle='dashed',label="Validation Loss")
    # Beautification of plot
    #ax.xaxis.set_major_locator(mdates.DayLocator(interval=1)) ####
    #ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
    ax.legend()
    ax.set_xlabel('Epochs', fontsize=15)
    #ax.set_ylabel('NO2 (mol/m2)', fontsize=15).
    ax.set_ylabel("Loss", fontsize=15)
    plt.setp(plt.gca().xaxis.get_majorticklabels(),'rotation', 45);
    ax.set_xlim([history_df.index[0], history_df.index[-1]])
    # Set the axis limits
    ax.set_ylim(0, 0.8)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.title(Title,size=20)
    plt.show()
    #ax.set_ylim(-0.2, 2.2)
    #plt.savefig(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/train_validation_loss{model_name}.png')

# Create figure object and store it in a variable called 'fig'
def test_prediction_plot(test_df,predicted,Title="Test vs Prediction (Model_name)"):
    fig, ax = plt.subplots(figsize = (14,6))
    # Edit the major and minor ticks of the x and y
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in', right='on')
    ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in', right='on')
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
    ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in', top='on')
    # plotting daily data
    # see: https://e2eml.school/matplotlib_lines.html for line colours
    ax.plot(test_df.index, test_df[test_df.columns[0]], color='black', label="Actual")
    # plotting 4-day rolling data
    ax.plot(test_df.index, predicted, color='red', linestyle='dashed',label="Predicted")
    # Beautification of plot
    #ax.xaxis.set_major_locator(mdates.DayLocator(interval=1)) ####
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%y'))
    ax.legend()
    ax.set_xlabel('Day', fontsize=15)
    #ax.set_ylabel('NO2 (mol/m2)', fontsize=15)
    ax.set_ylabel("Daily Confirmed Cases", fontsize=15)
    plt.setp(plt.gca().xaxis.get_majorticklabels(),'rotation', 45);
    ax.set_xlim([test_df.index[0], test_df.index[-1]])
    # Set the axis limits
    ax.set_ylim(-20, 6000)
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.show()
    #ax.set_ylim(-0.2, 2.2)
    plt.savefig(f'/content/drive/MyDrive/Colab Notebooks/Decomp_LSTM/Results/test_prediction_{model_name}.png')

def evaluation(true,predicted,model_name = '',prop=''):
    score_RMSE = math.sqrt(mean_squared_error(true, predicted))
    DS = direc_sim(true, predicted)
    linear_model = linear_regression_fit(predicted,true) # pred-true prediction assessment
    r_squared = round(linear_model.rsquared,2)
    results =  np.array([score_RMSE,DS,r_squared])
    columns = [model_name]
    dataframe = pd.DataFrame(data = results,columns = columns ,index=[f'RMSE_at_{prop}',f'DS_at_{prop}',f'r_squared_at_{prop}'])
    return dataframe
    