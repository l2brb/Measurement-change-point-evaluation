import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
import sys
import pm4py
import os
import re

# WRITE TIMESTAMPS BY FIRST CREATING A LIST WITH FILE NAMES ENDING IN .xes

directory_path = "/Users/luca/Documents/phd_things/tests_drift_measurement/Italian/sublog"
xes_files = [f for f in os.listdir(directory_path) if f.endswith('.xes')]

# SORT BY CREATION DATE
xes_files.sort(key=lambda f: os.path.getctime(os.path.join(directory_path, f)))

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
      return [ atoi(c) for c in re.split(r'(\d+)', text) ]
xes_files.sort(key=natural_keys)


df_timestamp=[]

for i in range(len(xes_files)):
    log = pm4py.read_xes('/Users/luca/Documents/phd_things/tests_drift_measurement/Italian/sublog/'+xes_files[i])
    print(log[0][0]['time:timestamp'])
    df_timestamp.append(log[0][0]['time:timestamp'].strftime("%Y/%m/%d"))

c=0
l=[]
for el in df_timestamp:
    if c%3==0:
        l.append(el)
    else:
        l.append(c*" ")
    c+=1



# NORMALIZE THE VALUES OF THE MEASURES

INFTY = 100000000
MEASURES_MAX_MIN = {
    "Support": [0, 1],
    "Confidence": [0, 1],
    "Recall": [0, 1],
    "Lovinger": [-INFTY, 1],
    "Specificity": [0, 1],
    "Accuracy": [0, 1],
    "Lift": [0, INFTY],
    "Leverage": [-1, 1],
    "Compliance": [0, 1],
    "Odds Ratio": [0, INFTY],
    "Gini Index": [0, 1],
    "Certainty factor": [-1, 1],
    "Coverage": [0, 1],
    "Prevalence": [0, 1],
    "Added Value": [-1, 1],
    "Relative Risk": [0, INFTY],
    "Jaccard": [0, 1],
    "Ylue Q": [-1, 1],
    "Ylue Y": [-1, 1],
    "Klosgen": [-1, 1],
    "Conviction": [0, INFTY],
    "Interestingness Weighting Dependency": [0, INFTY],
    "Collective Strength": [-INFTY, INFTY],
    "Laplace Correction": [0, 1],
    "J Measure": [-INFTY, INFTY],
    "One-way Support": [-INFTY, INFTY],
    "Two-way Support": [-INFTY, INFTY],
    "Two-way Support Variation": [-INFTY, INFTY],
    "Linear Correlation Coefficient": [-INFTY, INFTY],
    "Piatetsky-Shapiro": [-1, 1],
    "Cosine": [0, 1],
    "Information Gain": [-INFTY, INFTY],
    "Sebag-Schoenauer": [0, INFTY],
    "Least Contradiction": [-INFTY, INFTY],
    "Odd Multiplier": [0, INFTY],
    "Example and Counterexample Rate": [-INFTY, 1],
    "Zhang": [-INFTY, INFTY]
}


def normalize_single_measure(x, measure):
    """
        ((x / (1 + abs(x)) – min_norm )/(max_norm – min_norm)

        min_norm= min /( 1+ abs(min) )
        max_norm= max /( 1+ abs(max) )

        infinity approximated to 100000000

    :param x: measure measurement
    :param measure: measure identifyier
    :return normalized measure
    """
    min_norm = MEASURES_MAX_MIN[measure][0] / (1 + abs(MEASURES_MAX_MIN[measure][0]))
    max_norm = MEASURES_MAX_MIN[measure][1] / (1 + abs(MEASURES_MAX_MIN[measure][1]))

    return (x / (1 + abs(x)) - min_norm) / (max_norm - min_norm)


def normalize_measure_trend(vector, measure):
    for i in range(len(vector)):
        vector[i] = normalize_single_measure(vector[i], measure)
    return vector


# MEASURES DATASET IMPORT

df = pd.read_csv('sublog_measures_italian.csv', sep=';', names=['SUBLOG', 'Support',	'Confidence',	'Recall',	'Lovinger',	'Specificity',	'Accuracy',	'Lift',	'Leverage',	'Compliance',	'Odds Ratio',	'Gini Index',	'Certainty factor',	'Coverage',	'Prevalence',	'Added Value',	'Relative Risk',	'Jaccard',	'Ylue Q',	'Ylue Y',	'Klosgen', 'Conviction',	'Interestingness Weighting Dependency',	'Collective Strength',	'Laplace Correction',	'J Measure',	'One-way Support',	'Two-way Support',	'Two-way Support Variation',	'Linear Correlation Coefficient',	'Piatetsky-Shapiro',	'Cosine',	'Information Gain',	'Sebag-Schoenauer',	'Least Contradiction',	'Odd Multiplier',	'Example and Counterexample Rate',	'Zhang'], header=0)

# DATASET CLEANING

df[['SUBLOG', 'SUBLOG_2']] = df.SUBLOG.str.split("-", expand = True)
df['SUBLOG'] = df['SUBLOG'].astype(int)
df = df.sort_values('SUBLOG').reset_index(drop=True)
df['SUBLOG'] = df['SUBLOG'].astype(str) + '-' + df['SUBLOG_2']


"""Normalization test

Lovinger Example
test_lovinger = (df['Lovinger'].tolist())
print(test_lovinger)
print(normalize_measure_trend(test_lovinger, measure='Lovinger'))
print()"""



list_vectors = []
for i in range(len(MEASURES_MAX_MIN)):
    list_vectors.append(df[list(MEASURES_MAX_MIN.keys())[i]].tolist())


vector_normalized = []

for i in range(len(MEASURES_MAX_MIN)):
    vector_normalized.append(normalize_measure_trend(list_vectors[i], measure=list(MEASURES_MAX_MIN.keys())[i]))

df_normalized = pd.DataFrame(vector_normalized)
df_normalized = df_normalized.transpose()
df_normalized.columns = list(MEASURES_MAX_MIN.keys())


# INSERT THE SUBLOG COLUMN FROM THE ORIGINAL DATASET

sublist = df['SUBLOG'].tolist()

df_normalized.insert(loc=0, column='SUBLOG', value=sublist)
df_normalized.insert(loc=0, column='TIMESTAMP', value=l)
# df_normalized.to_csv('valori_normal_timest.csv')



# CREATE PLOT [NORMALIZED VALUED]

plt.style.use("seaborn-v0_8-bright")
graph = plt.figure(figsize=(16,9))

plt.plot(df_normalized['TIMESTAMP'], df_normalized['Recall'], color="violet", label="Recall", linewidth=0.5, linestyle="-", marker=".", markersize=4)
plt.plot(df_normalized['TIMESTAMP'], df_normalized['Confidence'], color="green", label="Confidence", linewidth=1, linestyle="-", marker="H", markersize=4)
plt.plot(df_normalized['TIMESTAMP'], df_normalized['Klosgen'], color="orange", label="Klosgen", linewidth=1, linestyle="-", marker="x", markersize=4)
plt.plot(df_normalized['TIMESTAMP'], df_normalized['Cosine'], color="red", label="Cosine", linewidth=1, linestyle="-", marker="<", markersize=4)
plt.plot(df_normalized['TIMESTAMP'], df_normalized['Least Contradiction'], color="gray", label="Least Contradiction", linewidth=1, linestyle="-", marker=">", markersize=4)
plt.plot(df_normalized['TIMESTAMP'], df_normalized['Sebag-Schoenauer'], color="darkblue", label="Sebag-Schoenauer", linewidth=1, linestyle="-", marker="8", markersize=4)
plt.plot(df_normalized['TIMESTAMP'], df_normalized['Laplace Correction'], color="purple", label="Laplace Correction", linewidth=1, linestyle="-", marker="X", markersize=4)
plt.plot(df_normalized['TIMESTAMP'], df_normalized['Interestingness Weighting Dependency'], color="lightblue", label="Interestingness Weighting Dependency", linewidth=1, linestyle="-", marker="p", markersize=4)

plt.grid('on', linestyle='--')
plt.title("ITALIAN MEASURES TRENDS")
plt.xticks(df_normalized['TIMESTAMP'], rotation=90,  fontsize=12)
plt.yticks(fontsize=15)

leg = plt.legend (loc='lower right', framealpha=1, edgecolor="black", fancybox=False)

graph.subplots_adjust(bottom=0.2)

graph.savefig('italian_measures_trend.pdf')
plt.show()
exit()


