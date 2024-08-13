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
dense_vars = ['varZ_det']
dense_var_to_use = dense_vars[0]
interesting_vars = hps_jet_vars + associatedjet_vars + others + [dense_var_to_use]

remove_vars = ['HPSJet_photonsOutsideSignalCone','Associatedjet_chEmEF','Associatedjet_muEF']#['HPSJet_leadTkDeltaR','Associatedjet_muEF','Associatedfatjet_tau2bytau1','Associatedfatjet_tau3bytau2','Associatedjet_chEmEF']

interesting_vars = [x for x in interesting_vars if x not in remove_vars]

dense_cut_values = {
        #'var_name' : [cut_value,'cut_string']
        'varZ_det': {'value':0,'str':'0p0'},#exploring fit->so no cut needed
}

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
        "dphi_HPSJet_recoil":{
                "min": 1,
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
        dense_var_to_use:{# no scaling the dense score
                "min": 0,
                "max":1,
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
X_train, X_val_test, Y_train, Y_val_test = train_test_split(X, Y, test_size=0.4, random_state=7)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5, random_state=7)
############# Data preprocessing #####################

# already done using scale_df 

######################## Define the Model ######################
import xgboost
# Create parameter grid
parameters_test = {"learning_rate":[0.1],"n_estimators":[5]}
parameters = {"learning_rate": [0.1, 0.01, 0.001],
               "gamma" : [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],
               "max_depth": [2, 4, 7, 10],
               "colsample_bytree": [0.3, 0.6, 0.8, 1.0],
               "subsample": [0.2, 0.4, 0.5, 0.6, 0.7],
               "reg_alpha": [0, 0.5, 1],
               "reg_lambda": [1, 1.5, 2, 3, 4.5],
               "min_child_weight": [1, 3, 5, 7],
               "n_estimators": [10,100, 250, 500, 1000]}

# create the model
## XGBRegressor
from sklearn.metrics import make_scorer,mean_squared_error

xgb_reg = xgboost.XGBRegressor(verbosity = 2, seed =0, eval_metric = 'rmse', objective = "reg:logistic")    
neg_mean_squared_error_scorer = make_scorer(mean_squared_error,greater_is_better=False)#https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_scorer.py#L624                      
xgb_rscv = RandomizedSearchCV(xgb_reg, param_distributions = parameters, scoring = neg_mean_squared_error_scorer,
                             cv = 7, verbose = 2, random_state = 40)

################### Run training ########################
print("#########Starting training########")
# Fit the model
model = xgb_rscv.fit(X_train, Y_train)
#### Save the model or load preexisting model ##############
#'''
import pickle # import joblib
# save the model to disk
filename_rsc = './xgb_%s_gre_%s_rsc_nocut.sav'%(dense_var_to_use,dense_cut_values[dense_var_to_use]['str'])
pickle.dump(model, open(filename_rsc, 'wb'))

# load the model from disk#
#model = pickle.load(open(filename, 'rb'))
#model_rsc = pickle.load(open(filename_rsc, 'rb'))
#model_gsc = pickle.load(open(filename_gsc, 'rb'))
#print(model.cv_results_['params'])
#print(model_rsc.best_estimator_)
#print(model_gsc.best_estimator_)
#input("done printing best esitmator?")
#'''

##################### Print performance ###################
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print(y_pred)

predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy is %.2f%%" % (accuracy * 100.0))
#input("aha?")
######################## Plot Performance ##########################

########################### Testing performance 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
######Partition the test set into sig/bkg sets

print("Y_test.shape: " + str(Y_test.shape))
test_sig_number = np.count_nonzero(Y_test)
test_bkg_number = np.count_nonzero(Y_test == 0)

print("test_sig_number=%s"%test_sig_number)
print("test_bkg_number=%s"%test_bkg_number)

X_test_sig = np.zeros((test_sig_number,X_test.shape[1]))
X_test_bkg = np.zeros((test_bkg_number,X_test.shape[1]))

test_df_scaled_sig = test_df_scaled[test_df_scaled['isSignal']==1]
test_df_scaled_bkg = test_df_scaled[test_df_scaled['isSignal']==0]
X_test_sig = test_df_scaled_sig.values[:,0:NDIM]
X_test_bkg = test_df_scaled_bkg.values[:,0:NDIM] 
######## Make plots

Y_predict = model.predict(X_test)
Y_predict_bkg = model.predict(X_test_bkg)
Y_predict_sig = model.predict(X_test_sig)
fpr, tpr, thresholds = roc_curve(Y_test, Y_predict)
roc_auc = auc(fpr, tpr)

img_test , axes = plt.subplots(2,1,figsize = (10,10),dpi=100)
#img_test = plt.figure(figsize = (20,20), dpi=100)
#axes = plt.subplot(2, 2, 1)

axes[0].hist(Y_predict_bkg, density = 1, range = (0.0, 1), bins = 10, alpha = 0.3, label = 'Bkg') # density is a newer argument, for older matplotlib versions use normed instead
axes[0].hist(Y_predict_sig, density = 1, range = (0.0, 1), bins = 10, alpha = 0.3, label = 'Zprime')
axes[0].legend(loc = 'upper center')
axes[0].set_title('Model prediction on test set (unbias), normalized')
axes[0].set_xlabel('Predicted probability of being a signal event')
axes[0].set_xticks(np.arange(0, 1.1, step = 0.1))

axes[1].plot(fpr, tpr, lw = 2, color = 'red', label = 'ROC curve for test set, AUC (area) = %.3f' % (roc_auc))
axes[1].plot([0, 1], [0, 1], linestyle = '--', lw = 2, color = 'black', label = 'random chance')
axes[1].set_xlim([0, 1.0])
axes[1].set_ylim([0, 1.0])
axes[1].set_xlabel('False positive rate (FPR)')
axes[1].set_ylabel('True positive rate (TPR)')
axes[1].set_title('Receiver operating characteristic (ROC)')
axes[1].legend(loc = "lower right")

plt.tight_layout()

img_test.savefig('xgboost_rsc_test_%s_gre_%s_nocut.png'%(dense_var_to_use,dense_cut_values[dense_var_to_use]['str']))
###################Feature importance 
from sklearn.inspection import permutation_importance
print("Y_val.shape: " + str(Y_val.shape))
permu_imp = permutation_importance(model, X_val, Y_val, n_repeats=100, random_state=7, scoring='roc_auc')
for i in permu_imp.importances_mean.argsort()[::-1]:
    print(f"{interesting_vars[i]}"
          f"{permu_imp.importances_mean[i]:.3f}"
          f" +/- {permu_imp.importances_std[i]:.3f}")


