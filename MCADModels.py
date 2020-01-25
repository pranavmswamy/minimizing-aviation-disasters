import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Loading the Dataset
train_df = pd.read_csv('train.csv')
#loft_df = pd.read_csv('test.csv')
#------------------------------------------------------------------------------------------
#Exploratory Data Analysis

# 1. Distribution of states of the pilot:
plt.figure(figsize=(7,12))
sns.countplot(train_df['event'], palette = 'pastel')
plt.xlabel("State of the Pilot", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Distribution of State of the Pilot", fontsize=15)
plt.show()

# 2. Time distribution of each event
plt.figure(figsize=(15,10))
sns.violinplot(x='event', y='time', data=train_df, palette = 'pastel')
plt.ylabel("Time (s)", fontsize=12)
plt.xlabel("Event", fontsize=12)
plt.title("When do the events occur?", fontsize=15)
plt.show()

#3. Distribution of ecg recordings
plt.figure(figsize=(15,10))
sns.distplot(loft_df['ecg'], label='Test set', bins = 120)
sns.distplot(train_df["ecg"], label='Train set', bins = 120)
plt.legend()
plt.xlabel("Electrocardiogram Signal (µV)", fontsize=12)
plt.title("Electrocardiogram Signal Distribution", fontsize=15)
plt.show()

#4. Distribution of respiration recordings
plt.figure(figsize=(15,10))
sns.distplot(loft_df['r'], label='Test set', bins=500)
sns.distplot(train_df['r'], label='Train set', bins = 500)
plt.legend()
plt.xlabel("Respiration Signal (µV)", fontsize=12)
plt.title("Respiration Signal Distribution", fontsize=15)
plt.show()

#5. Distribution of GSR recordings
plt.figure(figsize=(15,10))
sns.distplot(loft_df['gsr'], label='Test set')
sns.distplot(train_df['gsr'], label='Train set')
plt.legend()
plt.xlabel("Electrodermal activity measure (µV)", fontsize=12)
plt.title("Electrodermal activity Distribution", fontsize=15)
plt.show()
#------------------------------------------------------------------------------------------
#Data Preprocessing

features = ["eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6", "eeg_t5", "eeg_t3", "eeg_fp2", "eeg_o1", "eeg_p3", "eeg_pz", "eeg_f3", "eeg_fz", "eeg_f4", "eeg_c4", "eeg_p4", "eeg_poz", "eeg_c3", "eeg_cz", "eeg_o2", "ecg", "r", "gsr"]

train_df['pilot'] = 100 * train_df['seat'] + train_df['crew']

from sklearn.preprocessing import MinMaxScaler

pilots = train_df["pilot"].unique()
for pilot in pilots:
    ids = train_df[train_df["pilot"] == pilot].index
    scaler = MinMaxScaler()
    train_df.loc[ids, features] = scaler.fit_transform(train_df.loc[ids, features])

#------------------------------------------------------------------------------------------
#Splitting Training Data into Train Set and Test Set
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(train_df, test_size=0.2)

#------------------------------------------------------------------------------------------
# Building Models
# 1. Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
classifierDT = DecisionTreeClassifier()
classifierDT.fit(train_df[features],train_df["event"])

#2. Random Forest - 10 trees
from sklearn.ensemble import RandomForestClassifier
classifierRF10 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
classifierRF10.fit(train_df[features],train_df["event"])

#3. Random Forest - 50 trees
from sklearn.ensemble import RandomForestClassifier
classifierRF50 = RandomForestClassifier(n_estimators = 50, criterion = 'entropy')
classifierRF50.fit(train_df[features],train_df["event"])
#------------------------------------------------------------------------------------------
# Predicting Test set results

y_pred = classifierDT.predict(test_df[features])
y_pred = classifierRF10.predict(test_df[features])
y_pred = classifierRF50.predict(test_df[features])

#------------------------------------------------------------------------------------------
# Confusion Matrix and Accuracy Score
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_df["event"].values, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#print(cm)
#------------------------------------------------------------------------------------------


