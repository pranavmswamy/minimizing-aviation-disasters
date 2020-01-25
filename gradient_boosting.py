# -*- coding: utf-8 -*-
"""
Created on Mon May 27 17:32:53 2019

@author: Pranav
"""


# 4. Gradient Boosting - LGBM
import lightgbm as lgb

def run_lgb(df_train, df_test):
    dic = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    try:
        df_train["event"] = df_train["event"].apply(lambda x: dic[x])
        df_test["event"] = df_test["event"].apply(lambda x: dic[x])
    except: 
        pass
    
    params = {"objective" : "multiclass",
              "num_class": 4,
              "metric" : "multi_error",
              "num_leaves" : 30,
              "min_child_weight" : 50,
              "learning_rate" : 0.1,
              "bagging_fraction" : 0.7,
              "feature_fraction" : 0.7,
              "bagging_seed" : 420,
              "verbosity" : -1
             }
    
    lg_train = lgb.Dataset(df_train[features], label=(df_train["event"]))
    lg_test = lgb.Dataset(df_test[features], label=(df_test["event"]))
    model = lgb.train(params, lg_train, 1000, valid_sets=[lg_test], early_stopping_rounds=50, verbose_eval=100)
    
    return model

model = run_lgb(train_df, test_df)



pred_val = model.predict(test_df[features], num_iteration=model.best_iteration)


cf_mt = confusion_matrix(np.argmax(pred_val, axis=1), test_df["event"].values)
cf_mt = cf_mt.astype('float') / cf_mt.sum(axis=1)[:, np.newaxis]



# 3. Distribution of eeg recordings

plt.figure(figsize=(20,25))
plt.title('Eeg features distributions')
i = 0
eeg_recordings = ["eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6", "eeg_t5", "eeg_t3", "eeg_fp2", "eeg_o1", "eeg_p3", "eeg_pz", "eeg_f3", "eeg_fz", "eeg_f4", "eeg_c4", "eeg_p4", "eeg_poz", "eeg_c3", "eeg_cz", "eeg_o2"]
for eeg in eeg_recordings:
    i += 1
    plt.subplot(5, 4, i)
    sns.distplot(loft_df[eeg], label='Test set', hist=True)
    sns.distplot(train_df[eeg], label='Train set', hist=True)
    plt.legend()
    plt.xlabel(eeg, fontsize=12)

plt.show()
