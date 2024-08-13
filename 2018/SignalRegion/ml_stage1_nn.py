from __future__ import division
import os
import uproot
import numpy as np
import pandas as pd
import h5py

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

treename = 'Events'
filenames = {}
upfile = {}
params = {}
df = {}
df_scaled = {}
available_events = {}
events_to_select = {}
input_path_base = './ml_input_files/'

filenames['Zprime'] = input_path_sig + 'all_monoZprime.root'
#filenames['bkg'] = input_path_bkg + 'all_bkg_samples.root'
#'''
filenames['DYJets'] = input_path_bkg + 'all_DYJets.root'
#filenames['DY2Jets'] = input_path_bkg + 'all_DY2Jets.root'
filenames['Z1Jets'] = input_path_bkg + 'all_Z1Jets.root'
filenames['Z2Jets'] = input_path_bkg + 'all_Z2Jets.root'
filenames['WJets'] = input_path_bkg + 'all_WJets.root'
filenames['GJets'] = input_path_bkg + 'all_GJets.root'
filenames['QCD'] = input_path_bkg + 'all_QCD.root'
filenames['Top'] = input_path_bkg +  'all_Top.root'
filenames['DiBoson'] = input_path_bkg + 'all_DiBoson.root'

hps_jet_vars = ['HPSJet_photonsOutsideSignalCone','HPSJet_leadTkPtOverhpsPt','HPSJet_leadTkDeltaR','HPSJet_mass','HPSJet_decaymode'] # HPSJet
associatedjet_vars = ['leadTkPtOverAssociatedJetPt','HPSJetPtOverAssociatedJetPt','HPSJet_AssociatedJet_DeltaR','Associatedjet_chHEF','Associatedjet_chEmEF','Associatedjet_neEmEF','Associatedjet_neHEF','Associatedjet_muEF','Associatedjet_mass'] #AssociatedJet
others= ['dphi_HPSJet_recoil','recoil'] #others
interesting_vars = hps_jet_vars + associatedjet_vars + others 

remove_vars = ['HPSJet_photonsOutsideSignalCone','Associatedjet_chEmEF','Associatedjet_muEF']

interesting_vars = [x for x in interesting_vars if x not in remove_vars]

min_max_dict = {
        "HPSJet_photonsOutsideSignalCone":{
                "min": 0,
                "max":30,
        },
        "HPSJet_leadTkPtOverhpsPt":{
                "min": 0,
                "max":2,
        },
        "HPSJet_leadTkDeltaR":{
                "min": 0,
                "max":0.1,
        },
        "HPSJet_mass":{
                "min": 0,
                "max":5,
        },
        "leadTkPtOverAssociatedJetPt":{
                "min": 0,
                "max":1,
        },
        "HPSJetPtOverAssociatedJetPt":{
                "min": 0,
                "max":1.5,
        },        
        "HPSJet_AssociatedJet_DeltaR":{
                "min": 0,
                "max":0.2,
        },
        "Associatedjet_chHEF":{
                "min": 0,
                "max":1,
        },
        "Associatedjet_chEmEF":{
                "min": 0,
                "max":1,
        },
        "Associatedjet_neEmEF":{
                "min": 0,
                "max":1,
        },
        "Associatedjet_neHEF":{
                "min": 0,
                "max":1,
        },
        "Associatedjet_muEF":{
                "min": 0,
                "max":1,
        },
        "Associatedjet_mass":{
                "min": 0,
                "max":150,
        },
        "dphi_HPSJet_recoil":{#dphi_HPSJet_MET
                "min": 1.0,#0.5,
                "max":3.2,
        },
        "recoil":{
                "min": 250,
                "max":2000,
        },                
        'isSignal':{  # adding it so that df.subtraction doesn't give NaN values when scaling
                "min": 0.0,
                "max":1.0,
        },       
        "dense23":{# no scaling the dense score
                "min": 0,
                "max":1,# some mean values >1!, setting values above 1 to be 1 below
        },    
        "HPSJet_idDeepTau2017v2p1VSjet":{
                #"min": 2**1,#only choosing events where HPSJet passes VVLoose DeepTau id
                "min": 2**2,#only choosing events where HPSJet passes VLoose DeepTau id
                "max":300,#max value is 255
        },    
        "HPSJet_decaymode":{
                "min": 0,
                "max":12,
        },                                                                                                                                          
}
def get_from_file(root_file,tag):
        print("Getting from file: %s"%root_file)
        upfile[tag] = uproot.open(root_file)
        vars_to_get = interesting_vars + ['HPSJet_idDeepTau2017v2p1VSjet']
        df[tag] = upfile[tag][treename].arrays(vars_to_get, library="pd")# otherwise awkward used by default
        ### claning so that df only contains values within min and max of variables
        for var in vars_to_get:
                if(var in [dense_var_to_use]): continue # no cut on dense_var_to_use score
                var_cut = (df[tag][var] >= min_max_dict[var]['min']) & (df[tag][var] <= min_max_dict[var]['max'])
                df[tag] = df[tag][var_cut]
                #print(df[tag][df[tag][dense_var_to_use]>1].shape)
                #print(df[tag][df[tag][dense_var_to_use]<0].shape)
                print("df['%s'].shape after %s cleaning=%s"%(tag,var,df[tag].shape))   
        df[tag].drop('HPSJet_idDeepTau2017v2p1VSjet',axis=1,inplace=True)                     
        df[tag].loc[df[tag][dense_var_to_use]>1,dense_var_to_use] = 1 # setting values above 1 to be 1
        df[tag].loc[df[tag][dense_var_to_use]<0,dense_var_to_use] = 0 # setting values above 1 to be 1
        print("df['%s'].shape after %s cleaning of greater_than %s=%s"%(tag,dense_var_to_use,dense_cut_values[dense_var_to_use]['value'],df[tag].shape))
        #input("next?")
        available_events[tag] =  df[tag].shape[0] 


## preprocessing function
def scale_df(df,min_pd_series,max_pd_series):  
        df_scaled = df.subtract(min_pd_series,axis=1)/(max_pd_series-min_pd_series)
        #df_scaled_values = df_scaled.values
        return df_scaled
##

for key in filenames.keys(): 
        get_from_file(filenames[key],key)
        print("Got %s events from %s"%(df[key].shape[0],key)) 
print(available_events)
input("available events ok?")

## training selection
n_selections = 50000
bkg_fracs = { # fraction to use in training data
'Z1Jets':0.25,
'Z2Jets':0.25,
'WJets':0.4,
'QCD':0.05,
'Top':0.02,
'DYJets':0.01,
#'DY1Jets':0.005,
#'DY2Jets':0.005,
'GJets':0.01,
'DiBoson':0.01, 
}
for k in bkg_fracs.keys(): 
        #select_events = available_events[k]
        select_events = bkg_fracs[k]*n_selections
        if(select_events>available_events[k]): select_events = available_events[k]
        events_to_select[k] = int(select_events)
        print("%s:select_events=%s\n"%(k,select_events))
        print("%s:unblinded percent=%s\n"%(k,str((select_events/available_events[k])*100)))# don't want to look at all available events       

df['DYJets'] = df['DYJets'].sample(n = events_to_select['DYJets'], random_state = 2) 
df['GJets'] = df['GJets'].sample(n = events_to_select['GJets'], random_state = 2) 
df['WJets'] = df['WJets'].sample(n = events_to_select['WJets'], random_state = 2)
df['QCD'] = df['QCD'].sample(n = events_to_select['QCD'], random_state = 2) 
df['Top'] = df['Top'].sample(n = events_to_select['Top'], random_state = 2) 
df['Z1Jets'] = df['Z1Jets'].sample(n = events_to_select['Z1Jets'], random_state = 2)
df['Z2Jets'] = df['Z2Jets'].sample(n = events_to_select['Z2Jets'], random_state = 2)
df['DiBoson'] = df['DiBoson'].sample(n = events_to_select['DiBoson'], random_state = 2)
input("ok?")

unc_low = dict.fromkeys(interesting_vars,0.5); unc_low['recoil'] = 0.95; unc_low['HPSJet_decaymode'] = 1 ; unc_low[dense_var_to_use] = 1#std is very small 
unc_high = dict.fromkeys(interesting_vars,1.5); unc_high['recoil'] = 1.05; unc_high['HPSJet_decaymode'] = 1; unc_high[dense_var_to_use] = 1     
min_max_interesting = {}
for var in interesting_vars: min_max_interesting[var] = min_max_dict[var]
df_min_max = pd.DataFrame(min_max_interesting,index=["min","max"]).loc[:,interesting_vars]
#df_min_max['isSignal'] = pd.Series(min_max_dict['isSignal'],index=['min','max'])
min_pd_series = df_min_max.loc["min"].multiply(pd.Series(unc_low))
max_pd_series = df_min_max.loc["max"].multiply(pd.Series(unc_high)) 
print(min_pd_series)
print(max_pd_series)

for key in df.keys(): 
        print(key)
        df_scaled[key] = scale_df(df[key],min_pd_series,max_pd_series) # for now keeping the bounds same as before
        min_max_dict = {}
        min_max_dict['min'] = df_scaled[key].min()
        min_max_dict['max'] = df_scaled[key].max()
        min_max_df = pd.DataFrame(min_max_dict)
        print(min_max_df)
        #input('next?')



################## defining some useful variables #######################

NDIM = len(interesting_vars)


# add isSignal variable
for key in df_scaled.keys():
        if (key == 'Zprime'): df_scaled[key]['isSignal'] = np.ones(len(df_scaled[key])) 
        else: 
                df_scaled[key]['isSignal'] = np.zeros(len(df_scaled[key])) 

df_scaled['bkg'] = pd.concat([df_scaled[key] for key in df_scaled.keys() if key not in ['Zprime','data']])

# ############# Making the signal events equal to background events##########
sig_percent = (df_scaled['Zprime'].shape[0]/(df_scaled['Zprime'].shape[0]+df_scaled['bkg'].shape[0]))*100
print("sig_percent_before=%s\n"%sig_percent)
df_scaled['Zprime'] = df_scaled['Zprime'].sample(n = df_scaled['bkg'].shape[0] , random_state = 2)
#df['Zprime'] = df['Zprime'].sample(n = n_selections , random_state = 2)
sig_percent = (df_scaled['Zprime'].shape[0]/(df_scaled['Zprime'].shape[0]+df_scaled['bkg'].shape[0]))*100
print("sig_percent_after=%s\n"%sig_percent)   # should be 50% each

input("all ok?")
################## defining some useful variables #######################
df_all = pd.concat([df_scaled['Zprime'],df_scaled['bkg']])
dataset = df_all.values
print(type(dataset))
print(dataset)
print(len(dataset))
################# Getting the class weights--should be 1 if equal number of sig and bkg ###################
'''
# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_bkg = (1 / n_bkg)*(n_total)/2.0 
weight_for_sig = (1 / n_sig)*(n_total)/2.0

class_weight = {0: weight_for_bkg, 1: weight_for_sig}

print('Weight for class bkg: {:.2f}'.format(weight_for_bkg))
print('Weight for class sig: {:.2f}'.format(weight_for_sig))
'''
########### Dividing data into testing and training dataset ############

X = dataset[:,0:NDIM]
Y = dataset[:,NDIM]

from sklearn.model_selection import train_test_split
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

test_df_scaled = pd.DataFrame(X_test,columns=interesting_vars); test_df_scaled['isSignal'] = Y_test.tolist()#already scaled above
test_df_scaled_values = test_df_scaled.values
X_test = test_df_scaled_values[:,0:NDIM]; Y_test = test_df_scaled_values[:,NDIM]
############# Data preprocessing #####################

# already done using scale_df 

input("Start training?")
######################## Define the Model ######################
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
#from tensorflow.keras.utils import np_utils
import tensorflow.keras.metrics

inputs = Input(shape=(NDIM,), name = 'input')
middle_1 = Dense(units= 48, name = 'hidden_layer1', kernel_initializer='normal')(inputs)#
middle_1_prelu = tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(value=0.25))(middle_1)
dropout_1 = Dropout(rate = 0.3, seed = 10, name = 'Dropout_layer_1')(middle_1_prelu)#(middle_1)
middle_2 = Dense(units= 36, name = 'hidden_layer2', kernel_initializer='normal')(dropout_1)#(middle_1)#
middle_2_prelu = tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(value=0.25))(middle_2)
dropout_2 = Dropout(rate = 0.5, seed = 10, name = 'Dropout_layer_2')(middle_2_prelu)#(middle_2)
middle_3 = Dense(units= 3, name = 'hidden_layer3', kernel_initializer='normal')(dropout_2)
output = Dense(1, name = 'output', kernel_initializer='normal', activation='sigmoid')(middle_3)#det_nn
      
model = Model(inputs=inputs, outputs=output)

model.compile(optimizer='adam', run_eagerly=False, loss='binary_crossentropy', metrics=['accuracy']) 
model_name= 'varZ_det'

model.summary()        
batch_size = int(n_selections/10)
############ To avoid overfitting ##################
#'''
# early stopping callback
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# model checkpoint callback
# this saves our model architecture + parameters into dense_model.h5
from tensorflow.keras.callbacks import ModelCheckpoint
model_checkpoint = ModelCheckpoint('%s.h5'%model_name, monitor='val_loss', 
                                verbose=0, save_best_only=True, 
                                save_weights_only=False, mode='auto', 
                                period=1)

################### Run training ########################
print("#########Starting training########")
# Train classifier
history = model.fit(X_train_val, 
                Y_train_val, 
                epochs=10, 
                batch_size=,
                verbose=1, # switch to 1 for more verbosity 
                callbacks=[early_stopping, model_checkpoint], 
                validation_split=0.25)
#input("next?")
print(history.history.keys())
### Save history so that we can use for plotting later
import pickle
with open('./%s.pkl'%model_name, 'wb') as train_hist_pi: 
        pickle.dump(history.history, train_hist_pi)
        
######################## Plot Performance ##########################

########################### Training performance 
import matplotlib.pyplot as plt # no need as already using in correlation

#%matplotlib inline
# plot loss vs epoch
img_train = plt.figure(figsize=(15,10))
ax = plt.subplot(2, 1, 1)
ax.plot(history.history['loss'], label='loss')
ax.plot(history.history['val_loss'], label='val_loss')
ax.legend(loc="upper right")
ax.set_xlabel('epoch')
ax.set_ylabel('loss')

# plot accuracy vs epoch
ax = plt.subplot(2, 1, 2)
ax.plot(history.history['accuracy'], label='acc')
ax.plot(history.history['val_accuracy'], label='val_acc')
ax.legend(loc="upper left")
ax.set_xlabel('epoch')
ax.set_ylabel('acc')

img_train.savefig('./%s_training.png'%model_name)
#'''
'''
############# To load a previously saved model ################

# Comment out the sections on Defining the model, overfitting, training and train_plotting                                  

from tensorflow.keras.models import load_model 

model_rel_path = './'

model_name = 'varZ_det.h5'
model_str = model_rel_path + model_name
model = load_model(model_str, compile=True)

model.summary()
'''
##################################### Testing performance 

Y_predict_test =  model(X_test)
test_df_scaled_sig = test_df_scaled[test_df_scaled['isSignal']==1]
test_df_scaled_bkg = test_df_scaled[test_df_scaled['isSignal']==0]
X_test_sig = test_df_scaled_sig.values[:,0:NDIM]
X_test_bkg = test_df_scaled_bkg.values[:,0:NDIM] 

######## Make plots

from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(Y_test, Y_predict_test)
#fpr, tpr, thresholds = roc_curve(Y_test, Y_predict_mean)
roc_auc = auc(fpr, tpr)
#### To get the threshold closest to (0,1) from ROC curve
# calculate the g-mean for each threshold
gmeans = np.sqrt(tpr * (1-fpr))
# locate the index of the largest g-mean
ix = np.argmax(gmeans)
print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))

##### Plot other thresholds on the same ROC
thresh_to_plot = [0.7,0.9]
indices_to_plot = []
is_sorted = lambda a: np.all(a[:-1] >= a[1:])
print(is_sorted(thresholds))
for thresh in thresh_to_plot:
        difference_array = np.absolute(thresholds-thresh)
        indices_to_plot.append(difference_array.argmin())
for thresh_ind in indices_to_plot:
        print(thresholds[thresh_ind])
        print("-----")
                

img_test , axes = plt.subplots(2,1,figsize = (10,10),dpi=100)
#img_test = plt.figure(figsize = (20,20), dpi=100)
#axes = plt.subplot(2, 2, 1)

axes[0].hist(model.predict(X_test_bkg), density = 1, range = (0.0, 1.0), bins = 10, alpha = 0.3, label = 'Bkg') # density is a newer argument, for older matplotlib versions use normed instead
axes[0].hist(model.predict(X_test_sig), density = 1, range = (0.0, 1.0), bins = 10, alpha = 0.3, label = 'Zprime')  
axes[0].legend(loc = 'upper center')
axes[0].set_title('Model prediction on test set (unbias), normalized')
axes[0].set_xlabel('Predicted probability of being a signal event')
axes[0].set_xticks(np.arange(0, 1.1, step = 0.1))

axes[1].plot(fpr, tpr, lw = 2, color = 'red', label = 'ROC curve for test set, AUC (area) = %.3f' % (roc_auc))
axes[1].plot([0, 1], [0, 1], linestyle = '--', lw = 2, color = 'black', label = 'random chance')
axes[1].scatter(fpr[ix], tpr[ix], marker='o', color='black', label='Best thresold = %s'%thresholds[ix], s =100 )

i=0
for thresh_ind in indices_to_plot:
        if(i==0): axes[1].scatter(fpr[thresh_ind], tpr[thresh_ind], marker='v', color='violet', label='thresold = %s'%round(thresholds[thresh_ind],1), s =100) ; i+=1
        else: axes[1].scatter(fpr[thresh_ind], tpr[thresh_ind], marker='s', color='brown', label='thresold = %s'%round(thresholds[thresh_ind],1), s =100)
axes[1].set_xlim([0, 1.0])
axes[1].set_ylim([0, 1.0])
axes[1].set_xlabel('False positive rate (FPR)')
axes[1].set_ylabel('True positive rate (TPR)')
axes[1].set_title('Receiver operating characteristic (ROC)')
axes[1].legend(loc = "lower right")

plt.tight_layout()

img_test.savefig('%s_test_plots.png'%model_name)
