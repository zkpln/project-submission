#used code from a tutorial to get data processed

from __future__ import division, print_function, unicode_literals
#Load libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns
from scipy import stats
from scipy.stats import norm

#load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")



['LotFrontage',
 'Alley',
 'MasVnrType',
 'MasVnrArea',
 'BsmtQual',
 'BsmtCond',
 'BsmtExposure',
 'BsmtFinType1',
 'BsmtFinType2',
 'Electrical',
 'FireplaceQu',
 'GarageType',
 'GarageYrBlt',
 'GarageFinish',
 'GarageQual',
 'GarageCond',
 'PoolQC',
 'Fence',
 'MiscFeature']

#missing value counts in each of these columns
miss = train.isnull().sum()/len(train)
miss = miss[miss > 0]
miss.sort_values(inplace=True)
miss

#removing outliers
train.drop(train[train['GrLivArea'] > 4000].index, inplace=True)
train.shape #removed 4 rows`
(1456, 81)

#imputing using mode
test.loc[666, 'GarageQual'] = "TA" 
test.loc[666, 'GarageCond'] = "TA" 
test.loc[666, 'GarageFinish'] = "Unf" 
test.loc[666, 'GarageYrBlt'] = "1980"  

#mark as missing
test.loc[1116, 'GarageType'] = np.nan


#Modified Linear Model Code
from sklearn.linear_model import Lasso

#found this best alpha through cross-validation
best_alpha = 0.00099

regr = Lasso(alpha=best_alpha, max_iter=50000)







#  imports
import numpy as np
import os


np.random.seed(42)

# To plot 

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "training_linear_models"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

# Ignore warnings
import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")

from sklearn.linear_model import Ridge

np.random.seed(40)
m = 20
X = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
X_new = np.linspace(0, 3, 100).reshape(100, 1)




