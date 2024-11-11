# -*- coding: utf-8 -*-
"""
Judgment by Algorithm: Exploring AI Fairness in Criminal Justice
Synthetic Data Analysis

November 11, 2024
"""

# %% Modules

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %% Paths & Parameters

DIR = os.path.dirname(os.path.abspath(__file__)) # Update as necessary
os.chdir(DIR)

# Input directory with input data files
INPUT_DIR = DIR+'\\input'
INPUT_CSV_FILES = glob.glob(os.path.join(INPUT_DIR, '*.csv'))

# Output directory & file for output
OUTPUT_DIR = DIR+'\\output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
OUTPUT_EXCEL = OUTPUT_DIR+'\\Analysis_output.xlsx'
if not os.path.exists(OUTPUT_EXCEL):
    df = pd.DataFrame()
    df.to_excel(OUTPUT_EXCEL, index=False)


# %% Read data

df_master = pd.DataFrame()
for file in INPUT_CSV_FILES:
    df = pd.read_csv(file)
    df_master = pd.concat([df_master, df], ignore_index=True)

print(df_master.head())


# %% Utility Functions

# Export to excel
def excel_export (dataframe, file_path, sheetname):
    with pd.ExcelWriter(file_path, mode='a', if_sheet_exists='replace') as writer:
        dataframe.to_excel(writer, sheet_name=sheetname, index=False)
        print(sheetname + " exported to " + file_path + "\n")


# Generate and Export Table (does not return all dataframes in loops)
def table_generate_export (topic, dataframe, index, index2_loop, columns_loop, value, aggType):
    # Crosstab
    def crosstab_exe():
        if value is None:
            table = pd.crosstab(index=[dataframe[i] for i in index], columns=dataframe[c],
                                   margins=True, margins_name="Total")
        else:
            table = pd.crosstab(index=[dataframe[i] for i in index], columns=dataframe[c],
                                   values=dataframe[value], aggfunc=aggType,
                                   margins=True, margins_name="Total")
        table = table.reset_index()
        print(table.head())
        excel_export(table, OUTPUT_EXCEL, sheet)
        return table
    
    # Determine whether 2nd-layer index loop exists
    print('>>> ' + topic)
    for c in columns_loop:
        sheet = f"{topic}_{c}"
        ind = index.copy()
        
        if index2_loop != None:
            for j in index2_loop:
                sheet = f"{topic}_{j}"
                index = ind.copy()
                index.append(j)
                table = crosstab_exe()
        else:
            table = crosstab_exe()
    return table


# %% 1. Analyze the demographic profiles of three counties:
#    Evaluate and visualize the demographic characteristics (race, gender, and education) 
#       in each county using pivot tables (Excel) or Pandas (Python).
#    [In Excel] Deliverable: A summary of demographic data visualized through pie or bar charts 
#       and a comparison of racial and gender distributions across counties.

topic = 'Demo'
table_generate_export(topic, df_master, ['County'], None, ['Race', 'Gender', 'Education Level'], None, None)


# %% 2.	Evaluate risk scores across demographic groups:
#	 Analyze the distribution of risk scores by race and gender within each county.
#	 Deliverable: A summary table showing average risk scores for each demographic group, 
#       with accompanying visualizations (e.g., bar charts). 
#       Provide an analysis of how risk scores vary across counties and demographic groups.

topic = 'Risk'
table_generate_export(topic, df_master, ['County'], None, ['Race', 'Gender'], 'Risk Score', 'mean')


# %% 3.	Compare judge decisions to AI risk scores:
#	 Investigate the alignment between judges’ bail decisions and AI-generated risk scores 
#       across racial and gender groups.
#	 Deliverable: Cross-tabulate risk scores and judge decisions using stacked bar charts 
#       to evaluate consistency. Discuss patterns of bias in decision-making, 
#       highlighting any discrepancies between groups.

topic = 'Bail_Risk'
table_generate_export(topic, df_master, ['Judge Decision'], ['Race', 'Gender'], ['Risk Score'], None, None)


# %% 4.	Analyze re-offense rates and fairness metrics:
#	 Calculate re-offense rates and key fairness metrics, including False Positive Rates (FPR) 
#       and False Negative Rates (FNR), for each racial group.
#	 Deliverable: Bar charts comparing FPR and FNR across racial groups, 
#       accompanied by an interpretation of any disparities found.

topic = 'Fairness_Judge'
print('>>> ' + topic)

# Encode & Reanme: Judge Decision, Re-offense
df_master['Re-offense_en'] = df_master['Re-offense'].map({'Yes':1, 'No':0})

# Metrics parameters
group = 'Race'
cond = 'Re-offense'     # condition: Re-offense
pred = 'Judge Decision' # prediction: Judge Decision
cond_pos = 'Yes'        # Re-offense: Yes
cond_neg = 'No'         # Re-offense: No
pred_pos = 'Denied'     # Judge Decision: Denied (bail)
pred_neg = 'Granted'    # Judge Decision: Granted (bail)

# Define generated variable names
df_fairness = pd.DataFrame()
var = cond
group_cnt = f"{group} Total"                # Group: Race Total
cond_pos_sum = f"{group} {cond} Total"      # Condition pos: Race Re-offense Total
cond_neg_sum = f"{group} No {cond} Total"   # Condition neg: Race No Re-offense Total
cond_mean = f"{group} {cond} Rate"          # Condition pos share: Race Re-offense Rate

# Summary by group: total population, condition sum, condition mean
df_reoffense = df_master.groupby('Race')[cond+'_en'].agg(['count', 'sum', 'mean']).reset_index()
df_reoffense.rename(columns={'count': group_cnt, 'sum': cond_pos_sum, 'mean': cond_mean}, inplace=True)
df_reoffense[cond_neg_sum] = df_reoffense[group_cnt] - df_reoffense[cond_pos_sum]

# Fairness metrics
table = table_generate_export(topic, df_master, [group, pred], None, [cond], None, None)
table = table[[group, pred, cond_pos, cond_neg]]      # order columns [Positive, Negative]

# Merge metrics with summary
df_fairness = pd.merge(table, df_reoffense, on='Race', how='left')

# FPR & FNR
# (pred_pos ∩ cond_neg)/cond_neg_sum ==> ('Denied' ∩ 'No')/'No'
df_fairness['FPR'] = np.where(df_fairness[pred] == pred_pos,
                              df_fairness[cond_neg]/df_fairness[cond_neg_sum], None)
# (pred_neg ∩ cond_pos)/cond_pos_sum ==> ('Granted' ∩ 'Yes')/'Yes'
df_fairness['FNR'] = np.where(df_fairness[pred] == pred_neg, 
                              df_fairness[cond_pos]/df_fairness[cond_pos_sum], None)

# Export fairness data
excel_export(df_fairness, OUTPUT_EXCEL, topic)


# %% 3alt. Risk score distribution by county, judge's decision, and race/gender
topic = 'Bail_Risk_alt'
table_generate_export(topic, df_master, ['County', 'Judge Decision'], ['Race', 'Gender'], ['Risk Score'], None, None)


# %% 3alt.  Risk score distribution by re-offense and race
topic = 'Reoffense_Risk_alt'
table_generate_export(topic, df_master, ['Re-offense'], ['Race', 'Gender'], ['Risk Score'], None, None)


# %% Accuracy
topic = 'Fairness_Risk'
print ('>>> ' + topic)

# Encode & Reanme: Judge Decision, Re-offense
df_master['Re-offense_en'] = df_master['Re-offense'].map({'Yes':1, 'No':0})


# Metrics parameters
group = 'Race'
cond = 'Re-offense'     # condition: Re-offense
pred = 'AI Decision'    # prediction: AI Decision
cond_pos = 'Yes'        # Re-offense: Yes
cond_neg = 'No'         # Re-offense: No
pred_pos = 'Denied'     # Judge Decision: Denied (bail)
pred_neg = 'Granted'    # Judge Decision: Granted (bail)

df_fairness_ai = pd.DataFrame()

# Loop through each Risk Score threshold
for i in range(1, 11):
    df = df_master[[group, cond, 'Risk Score']].copy()
    df.loc[df['Risk Score']>= i, pred] = pred_pos
    df.loc[df['Risk Score'] < i, pred] = pred_neg
    
    outcome = 'Outcome'
    df['Outcome'] = None
    df.loc[(df[cond]==cond_pos) & (df[pred]==pred_pos), outcome] = "TP"
    df.loc[(df[cond]==cond_neg) & (df[pred]==pred_pos), outcome] = "FP"
    df.loc[(df[cond]==cond_pos) & (df[pred]==pred_neg), outcome] = "FN"
    df.loc[(df[cond]==cond_neg) & (df[pred]==pred_neg), outcome] = "TN"
    
    threshold = 'Risk Score Threshold'
    df_grouped = df.groupby([group, 'Outcome']).size().reset_index(name='count')
    df_grouped.insert(0, threshold, i)
    df_fairness_ai = pd.concat([df_fairness_ai, df_grouped], ignore_index=True)
    
    # Fill in missing outcomes for each racial group & threshold
    u_group = df_fairness_ai[group].unique()
    u_threshold = df_fairness_ai[threshold].unique()
    u_outcome = df_fairness_ai[outcome].unique()
    complete_index = pd.MultiIndex.from_product([u_group, u_threshold, u_outcome], names=[group, threshold, outcome])
    df_complete_index = pd.DataFrame(index=complete_index).reset_index()
    df_fairness_ai = pd.merge(df_complete_index, df_fairness_ai, on=[group, threshold, outcome], how='left')
    df_fairness_ai['count'] = df_fairness_ai['count'].fillna(0)
    
# Export
sheet = topic + '_' + group
excel_export(df_fairness_ai, OUTPUT_EXCEL, topic)





