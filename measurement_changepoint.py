import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
import sys
import pm4py
import os

directory_path = "/Users/luca/Documents/phd_things/tests_drift_measurement/Italian/sublog"
xes_files = [f for f in os.listdir(directory_path) if f.endswith('.xes')]

# ORDINO PER DATA DI CREAZIONE
xes_files.sort(key=lambda f: os.path.getctime(os.path.join(directory_path, f)))

print(xes_files)
print()
import re


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

xes_files.sort(key=natural_keys)
print(xes_files)


df_timestamp=[]

for i in range(len(xes_files)):
    log = pm4py.read_xes('/Users/luca/Documents/phd_things/tests_drift_measurement/Italian/sublog/'+xes_files[i])
    print(log[0][0]['time:timestamp'])


    df_timestamp.append(log[0][0]['time:timestamp'].strftime("%Y/%m/%d"))

    #print(i)

print(df_timestamp)

c=0
l=[]
for el in df_timestamp:
    if c%3==0:
        l.append(el)
    else:
        l.append(c*" ")
    c+=1

print(l)




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
    :param measure: measure identifyier as per SJ2T
    :return normalized measure
    """
    min_norm = MEASURES_MAX_MIN[measure][0] / (1 + abs(MEASURES_MAX_MIN[measure][0]))
    max_norm = MEASURES_MAX_MIN[measure][1] / (1 + abs(MEASURES_MAX_MIN[measure][1]))

    return (x / (1 + abs(x)) - min_norm) / (max_norm - min_norm)


def normalize_measure_trend(vector, measure):
    for i in range(len(vector)):
        vector[i] = normalize_single_measure(vector[i], measure)
    return vector


# IMPORTO IL DATASET

df = pd.read_csv('sublog_measures_italian.csv', sep=';', names=['SUBLOG', 'Support',	'Confidence',	'Recall',	'Lovinger',	'Specificity',	'Accuracy',	'Lift',	'Leverage',	'Compliance',	'Odds Ratio',	'Gini Index',	'Certainty factor',	'Coverage',	'Prevalence',	'Added Value',	'Relative Risk',	'Jaccard',	'Ylue Q',	'Ylue Y',	'Klosgen', 'Conviction',	'Interestingness Weighting Dependency',	'Collective Strength',	'Laplace Correction',	'J Measure',	'One-way Support',	'Two-way Support',	'Two-way Support Variation',	'Linear Correlation Coefficient',	'Piatetsky-Shapiro',	'Cosine',	'Information Gain',	'Sebag-Schoenauer',	'Least Contradiction',	'Odd Multiplier',	'Example and Counterexample Rate',	'Zhang'], header=0)
# pd.set_option('display.max_rows',300)

# PULISCO IL DATASET
# Split single column into two columns use ',' delimiter
df[['SUBLOG', 'SUBLOG_2']] = df.SUBLOG.str.split("-", expand = True)
df['SUBLOG'] = df['SUBLOG'].astype(int)
df = df.sort_values('SUBLOG').reset_index(drop=True)
df['SUBLOG'] = df['SUBLOG'].astype(str) + '-' + df['SUBLOG_2']

# Testo la normalizzazione di una misura

# ESEMPIO Lovinger
# test_lovinger = (df['Lovinger'].tolist())
# print(test_lovinger)
# print(normalize_measure_trend(test_lovinger, measure='Lovinger'))
# print()

# ESEMPIO Certainty factor
# test_certfact = (df['Certainty factor'].tolist())
# print(test_certfact)
# print(normalize_measure_trend(test_certfact, measure='Certainty factor'))
# print()

list_vectors = []
for i in range(len(MEASURES_MAX_MIN)):
    list_vectors.append(df[list(MEASURES_MAX_MIN.keys())[i]].tolist())
# print(list_vectors[0])
# print(list_vectors[10])
# print(df.to_string())
# exit()

vector_normalized = []

for i in range(len(MEASURES_MAX_MIN)):
    vector_normalized.append(normalize_measure_trend(list_vectors[i], measure=list(MEASURES_MAX_MIN.keys())[i]))
    # print(vector_normalized)
    # print(i)
# print('\n')
# print(list(MEASURES_MAX_MIN.keys()))

# test_support = (df['Support'].tolist())
# print(test_support)
# print(normalize_measure_trend(test_support, measure='Support'))
# print(vector_normalized)
# print(len(vector_normalized))
# print(len(list(MEASURES_MAX_MIN.keys())))


# columns = list(MEASURES_MAX_MIN.keys())
# data = np.array(vector_normalized)
df_normalized = pd.DataFrame(vector_normalized)
df_normalized = df_normalized.transpose()
df_normalized.columns = list(MEASURES_MAX_MIN.keys())
# print(df_normalized)


# Inserisco la colonna SUBLOG dal dataset originario

sublist = df['SUBLOG'].tolist()

df_normalized.insert(loc=0, column='SUBLOG', value=sublist)
#df_normalized.insert(loc=0, column='TIMESTAMP', value=df_timestamp)
df_normalized.insert(loc=0, column='TIMESTAMP', value=l)


# print(df_normalized.to_string())
# df_normalized.to_csv('valori_normal_timest.csv')

# print(df.to_string())


# CREO I PLOT [VALORI NORMALIZZATI]

plt.style.use("seaborn-v0_8-bright")
graph = plt.figure(figsize=(16,9))


#plt.plot(df_normalized['TIMESTAMP'], df_normalized['Support'], color="blue", label="Support", linewidth=2, linestyle="-", marker=".")
plt.plot(df_normalized['TIMESTAMP'], df_normalized['Recall'], color="violet", label="Recall", linewidth=0.5, linestyle="-", marker=".", markersize=4)
plt.plot(df_normalized['TIMESTAMP'], df_normalized['Confidence'], color="green", label="Confidence", linewidth=1, linestyle="-", marker="H", markersize=4)
plt.plot(df_normalized['TIMESTAMP'], df_normalized['Klosgen'], color="orange", label="Klosgen", linewidth=1, linestyle="-", marker="x", markersize=4)
plt.plot(df_normalized['TIMESTAMP'], df_normalized['Cosine'], color="red", label="Cosine", linewidth=1, linestyle="-", marker="<", markersize=4)
plt.plot(df_normalized['TIMESTAMP'], df_normalized['Least Contradiction'], color="gray", label="Least Contradiction", linewidth=1, linestyle="-", marker=">", markersize=4)
plt.plot(df_normalized['TIMESTAMP'], df_normalized['Sebag-Schoenauer'], color="darkblue", label="Sebag-Schoenauer", linewidth=1, linestyle="-", marker="8", markersize=4)
plt.plot(df_normalized['TIMESTAMP'], df_normalized['Laplace Correction'], color="purple", label="Laplace Correction", linewidth=1, linestyle="-", marker="X", markersize=4)
plt.plot(df_normalized['TIMESTAMP'], df_normalized['Interestingness Weighting Dependency'], color="lightblue", label="Interestingness Weighting Dependency", linewidth=1, linestyle="-", marker="p", markersize=4)
#plt.plot(df_normalized['TIMESTAMP'], df_normalized['Leverage'], color="blue", label="Leverage", linewidth=1, linestyle="-", marker="p")
#plt.plot(df_normalized['TIMESTAMP'], df_normalized['Recall'], color="blue", label="Recall", linewidth=1, linestyle="-", marker="p", markersize=4)

plt.grid('on', linestyle='--')
plt.title("ITALIAN MEASURES TRENDS")
plt.xticks(df_normalized['TIMESTAMP'], rotation=90,  fontsize=12)
plt.yticks(fontsize=15)

leg = plt.legend (loc='lower right', framealpha=1, edgecolor="black", fancybox=False)

graph.subplots_adjust(bottom=0.2)

graph.savefig('grafico_italian_measures_trend.pdf')
plt.show()
exit()


# CREO I PLOT

# plt.plot(df['SUBLOG'],df['Support'], color="blue", label="support", linewidth=2, marker=".")
plt.plot(df['SUBLOG'], df['Confidence'], color="green", label="confidence", linewidth=2, linestyle="--", marker=".")
# plt.plot(df['SUBLOG'],df['Gini Index'], color="green", label="Coverage", linewidth=0.5, linestyle="--", marker=".")
# plt.plot(df['SUBLOG'],df['Accuracy'], color="red", label="Accuracy", linewidth=0.5, linestyle="--", marker=".")
# plt.plot(df['SUBLOG'],df['Odds Ratio'], color="orange", label="Odds Ratio", linewidth=0.5, linestyle="--", marker=".")
# plt.plot(df['SUBLOG'],df['Laplace Correction'], color="purple", label="Laplace Correction", marker=".")
# plt.plot(df['SUBLOG'],df['Jaccard'], color="green", label="Laplace Correction", marker=".")
# plt.plot(df['SUBLOG'],df['Prevalence'], color="gray", label="Laplace Correction", marker=".")
# plt.plot(df['SUBLOG'],df['Jaccard'], color="green", label="Laplace Correction", marker=".")
plt.grid()
plt.title("SUPPORT")
plt.xticks(df['SUBLOG'],rotation=90)


plt.show()

"""

plt.xlabel("SUBLOG")
plt.ylabel("Support")
plt.zlabel("Confidence")
plt.legend()

indexes = np.arange(5)
width = 0.3
plt.bar(indexes,y, color="red", label="support", width=width)
plt.bar(indexes + width,z, label="confidence", width=width)
plt.title("Support&Confidence")
plt.xlabel("SUBLOG")
plt.ylabel("Support")
plt.zlabel("Confidence")
plt.legend()
plt.xticks(indexes+width/2, x)
"""""
