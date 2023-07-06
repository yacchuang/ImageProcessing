# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:19:05 2023

@author: Ya-Chen.Chuang
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_excel('C:/Users/ya-chen.chuang/Documents/ChiariSymptoms_combine.xlsx', sheet_name='Sheet1')

print(df.head())

for column in df[3:21]:
    df.loc[df[column]=="N", column]=0
    df.loc[df[column]=="Y", column]=1
    
symptoms = ["syringomyelia", "headache", "dizziness", "back/neck/ear pain", "numbness", "fatigue", "visual symptoms", "nausea and vomit", "seizures", "tinnitus and vertigo", "weakness", "gait and imbalance", "sensory changes", "spasm/hyperreflecxic/jerking movement", "urinary/bowel", "psychological symptoms"]
df_symptom = df[symptoms].dropna()
df_surgery = df["Surgery"].dropna()

wb = Workbook()
sheet1 = wb.add_sheet('symptoms')

sheet.write(0,:,df_symptom.loc[0,:])

