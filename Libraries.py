# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:55:01 2023

@author: IreneBetsy.D
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:55:01 2023

@author: IreneBetsy.D
"""

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from flask import Flask, render_template,request,json,jsonify  
import traceback
import time 
import os
from datetime import datetime
from itertools import cycle
import pickle
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix,roc_curve,auc,roc_auc_score
from sklearn.preprocessing import label_binarize, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
