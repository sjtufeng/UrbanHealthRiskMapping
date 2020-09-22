# %%
import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor


# %%
# ______________________________________________________________________________________________________________________
# --------------Helper functions--------------
def print_full(x):
    pd.set_option('display.max.rows', len(x))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    # pd.set_option('display.float_format', '{:20, .2f}'.format)
    pd.set_option('display.max_colwidth', -1)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    # pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')


def get_value_counts_from_dataframe(df, col, name_val=None, name_count='count'):
    """Return a dataframe listing the counts of unique values in a column of the dataframe.

    :param df: the input dataframe
    :param col: the name of the column to be analyzed
    :param name_val: the column name for the unique values in the new dataframe
    :param name_count: the column name for the counts of the unique values in the new dataframe
    :return: a dataframe listing the counts of unique values
    """
    if name_val is None:
        name_val = col
    return df[col].value_counts().rename_axis(name_val).reset_index(name=name_count)


def get_strings_containing_substring(substring, list_str):
    """Return a list of strings which contain the substring (case-insensitive)."""
    return [s for s in list_str if substring.lower() in s.lower()]


def get_strings_not_in_second_list(list_str_1, list_str_2):
    """Return a list of strings which are in *list_str_1* but not in *list_str_2*."""
    return [s for s in list_str_1 if s not in list_str_2]


PROJECT_ROOT_DIR = "."
CITY_NAME = 'LA'
CSV_PATH = os.path.join(PROJECT_ROOT_DIR, 'tables', CITY_NAME)
os.makedirs(CSV_PATH, exist_ok=True)


def save_csv(csv_file, csv_name, index=False):
    path = os.path.join(CSV_PATH, csv_name + ".csv")
    print("Saving csv file", csv_name)
    csv_file.to_csv(path, index=index)


# %%
# ______________________________________________________________________________________________________________________
# --------------Prepare data for machine learning--------------
dirname = r"C:\_Academic\UT_Research\UrbanInfoLab\project_311Calls\data"
# %%
filename_LA_311_agg_tract = os.path.join(dirname, 'machine_learning/ml_LA_311_agg_tract.csv')
df_LA_311_agg_tract = pd.read_csv(filename_LA_311_agg_tract, dtype={'tract2010': str})
# %%
ser_count311 = df_LA_311_agg_tract.drop(columns=['tract2010']).sum()
ser_count311.sort_values(ascending=False, inplace=True)
all_311_types = ser_count311.index.tolist()
# %%
# select the most frequent service request types based on the 80/20 rule
cut_index = ser_count311.cumsum().searchsorted(0.8 * ser_count311.sum())
top_311_types = ser_count311.iloc[:cut_index].index.tolist()
# %%
# feature engineering
c1 = ['Bulky Items', 'Illegal Dumping Pickup']
c2 = ['Graffiti Removal']

df_LA_311_agg_tract['agg_ALL'] = df_LA_311_agg_tract.loc[:, all_311_types].sum(axis=1)
df_LA_311_agg_tract['agg_dumping'] = df_LA_311_agg_tract.loc[:, c1].sum(axis=1)
df_LA_311_agg_tract['agg_graffiti'] = df_LA_311_agg_tract.loc[:, c2].sum(axis=1)

df_LA_311 = pd.concat([df_LA_311_agg_tract[['tract2010']], df_LA_311_agg_tract.loc[:, 'agg_ALL':]], axis=1)

# %%
filename_LA_SLD_agg_tract = os.path.join(dirname, 'machine_learning/ml_LA_SLD_agg_tract.csv')
df_LA_SLD_agg_tract = pd.read_csv(filename_LA_SLD_agg_tract, dtype={'tract_id': str})
# %%
cols_sld = ['tract_id', 'P_WRKAGE',
            'PCT_AO1', 'PCT_AO2P',
            'R_PCTLOWWAGE', 'E_PCTLOWWA',
            'D1A', 'D1B', 'D1C', 'D1D',
            'D1C5_Ret10', 'D1C5_Off10', 'D1C5_Ind10', 'D1C5_Svc10', 'D1C5_Ent10',
            'D2A_JPHH', 'D2b_E5Mix',
            'D2a_EpHHm', 'D2c_TrpMx1', 'D2c_TripEq',
            'D2a_WrkEmp', 'D2c_WrEmIx',
            'D3a', 'D3aao', 'D3amm', 'D3apo',
            'D3b', 'D3bao', 'D3bmm3', 'D3bmm4', 'D3bpo3', 'D3bpo4',
            'D4b025', 'D4b050', 'D4d']
df_LA_SLD_agg_tract = df_LA_SLD_agg_tract[cols_sld]
# %%
filename_LA_SVI = os.path.join(dirname, 'machine_learning/ml_LA_SVI.csv')
df_LA_SVI = pd.read_csv(filename_LA_SVI, dtype={'FIPS': str})
# %%
cols_svi = ['FIPS'] + [c for c in df_LA_SVI.columns if c.startswith('EP_')]
df_LA_SVI = df_LA_SVI[cols_svi]
# %%
filename_LA_CDC = os.path.join(dirname, 'machine_learning/ml_LA_CDC.csv')
df_LA_CDC = pd.read_csv(filename_LA_CDC, dtype={'TractFIPS': str})
# %%
health_vars = ['ARTHRITIS_CrudePrev', 'BPHIGH_CrudePrev',
               'CANCER_CrudePrev', 'CASTHMA_CrudePrev',
               'CHD_CrudePrev', 'COPD_CrudePrev',
               'DIABETES_CrudePrev', 'HIGHCHOL_CrudePrev',
               'KIDNEY_CrudePrev',
               'MHLTH_CrudePrev', 'PHLTH_CrudePrev',
               'STROKE_CrudePrev', 'TEETHLOST_CrudePrev',
               'BINGE_CrudePrev', 'CSMOKING_CrudePrev',
               'LPA_CrudePrev', 'OBESITY_CrudePrev',
               'SLEEP_CrudePrev']
# %%
# merge data
d1 = pd.merge(df_LA_CDC[['TractFIPS', 'Population2010'] + health_vars], df_LA_311, left_on='TractFIPS',
              right_on='tract2010', how='left').drop(columns='tract2010')
d2 = pd.merge(d1, df_LA_SLD_agg_tract, left_on='TractFIPS', right_on='tract_id', how='left').drop(
    columns='tract_id')
d3 = pd.merge(d2, df_LA_SVI, left_on='TractFIPS', right_on='FIPS', how='left').drop(columns='FIPS')
df_LA_all_data = d3
# %%
# adjust the 311 counts by population
df_LA_all_data_adj = df_LA_all_data.copy()
cols_311_types = df_LA_311.columns.to_list()
cols_311_types.remove('tract2010')
df_LA_all_data_adj[cols_311_types] = df_LA_all_data_adj[cols_311_types].div(
    df_LA_all_data_adj['Population2010'], axis=0)
# %%
# check null values
cols_nan = df_LA_all_data_adj.columns[(df_LA_all_data_adj.isnull().sum() != 0)].to_list()
df_LA_all_data_adj[cols_nan].isnull().sum()
# %%
df_LA_all_data_adj.dropna(subset=cols_nan, inplace=True)
# %%
df_LA_tracts = df_LA_all_data_adj[['TractFIPS']].copy()
df_LA_pop = df_LA_all_data_adj[['Population2010']].copy()
df_LA_all_data_adj.drop(columns=['Population2010', 'TractFIPS'], inplace=True)

# %%
# shorten column names (<= 10 characters)
col_name_map = {'agg_ALL': 'SR_ALL', 'agg_dumping': 'SR_DUMPING', 'agg_graffiti': 'SR_GRAFFIT',

                'P_WRKAGE': 'P_WRKAGE', 'PCT_AO1': 'P_AO1',
                'PCT_AO2P': 'P_AO2P', 'R_PCTLOWWAGE': 'P_LOWWAGEr', 'E_PCTLOWWA': 'P_LOWWAGEe',
                'D1A': 'D_HH', 'D1B': 'D_POP', 'D1C': 'D_EMP', 'D1D': 'D_HUEMP',
                'D1C5_Ret10': 'D_EMP_RET', 'D1C5_Off10': 'D_EMP_OFF', 'D1C5_Ind10': 'D_EMP_IND',
                'D1C5_Svc10': 'D_EMP_SVC', 'D1C5_Ent10': 'D_EMP_ENT', 'D2A_JPHH': 'JOBSPERHH',
                'D2b_E5Mix': 'EMPMIX', 'D2a_EpHHm': 'EMPHHMIX', 'D2c_TrpMx1': 'TRIPMIX',
                'D2c_TripEq': 'TRIPEQ', 'D2a_WrkEmp': 'WRKSPERJOB', 'D2c_WrEmIx': 'HHWRKJOBEQ',
                'D3a': 'D_RD', 'D3aao': 'D_RD_AO', 'D3amm': 'D_RD_MM', 'D3apo': 'D_RD_PO',
                'D3b': 'D_X_EXCLAO', 'D3bao': 'D_X_AO', 'D3bmm3': 'D_X_MM3', 'D3bmm4': 'D_X_MM4',
                'D3bpo3': 'D_X_PO3', 'D3bpo4': 'D_X_PO4', 'D4b025': 'P_EMP025',
                'D4b050': 'P_EMP050', 'D4d': 'D_TRANSIT',

                'EP_POV': 'P_POV', 'EP_UNEMP': 'P_UNEMP',
                'EP_PCI': 'PCI', 'EP_NOHSDP': 'P_NOHSDP', 'EP_AGE65': 'P_AGE65P',
                'EP_AGE17': 'P_AGE17M', 'EP_DISABL': 'P_DISABL', 'EP_SNGPNT': 'P_SNGPNT',
                'EP_MINRTY': 'P_MINRTY', 'EP_LIMENG': 'P_LIMENG', 'EP_MUNIT': 'P_MUNIT',
                'EP_MOBILE': 'P_MOBILE', 'EP_CROWD': 'P_CROWD', 'EP_NOVEH': 'P_NOVEH',
                'EP_GROUPQ': 'P_GROUPQ', 'EP_UNINSUR': 'P_UNINSUR',

                'ARTHRITIS_CrudePrev': 'ARTHRITIS', 'BPHIGH_CrudePrev': 'BPHIGH', 'CANCER_CrudePrev': 'CANCER',
                'CASTHMA_CrudePrev': 'CASTHMA', 'CHD_CrudePrev': 'CHD', 'COPD_CrudePrev': 'COPD',
                'DIABETES_CrudePrev': 'DIABETES', 'HIGHCHOL_CrudePrev': 'HIGHCHOL', 'KIDNEY_CrudePrev': 'KIDNEY',
                'MHLTH_CrudePrev': 'MHLTH', 'PHLTH_CrudePrev': 'PHLTH', 'STROKE_CrudePrev': 'STROKE',
                'TEETHLOST_CrudePrev': 'TEETHLOST', 'BINGE_CrudePrev': 'BINGE', 'CSMOKING_CrudePrev': 'CSMOKING',
                'LPA_CrudePrev': 'LPA', 'OBESITY_CrudePrev': 'OBESITY', 'SLEEP_CrudePrev': 'SLEEP'}

df_LA_all_data_adj.rename(columns=col_name_map, inplace=True)


# %%
# ______________________________________________________________________________________________________________________
# --------------Train and test machine learning models--------------
def display_scores(cv_scores):
    """Display scores along with their mean and standard deviation."""
    print("Scores:", cv_scores)
    print("Mean:", cv_scores.mean())
    print("Standard deviation:", cv_scores.std())


def run_default_model(model, train_x, train_y):
    """Run a machine learning model with the default setting and display a few results."""
    model.fit(train_x, train_y)
    some_data = train_x.iloc[-10:]
    some_labels = train_y.iloc[-10:]
    print('Predictions:', model.predict(some_data))
    print('Labels:', list(some_labels))


def display_cv_scores(model, train_x, train_y, scoring='neg_mean_squared_error', cv=10):
    """Display cross-validation scores."""
    scores = cross_val_score(model, train_x, train_y, scoring=scoring, cv=cv)
    model_rmse_scores = np.sqrt(-scores)
    display_scores(model_rmse_scores)


def get_mean_cv_score(model, train_x, train_y, scoring='neg_mean_squared_error', cv=10):
    """Get the mean cross-validation score."""
    scores = cross_val_score(model, train_x, train_y, scoring=scoring, cv=cv)
    model_rmse_scores = np.sqrt(-scores)
    return model_rmse_scores.mean()


def param_grid_search_lasso(train_x, train_y, max_iter=10000, cv=10, scoring='neg_mean_squared_error'):
    """Return the best Lasso regression model."""
    param_grid = {'alpha': np.arange(0.1, 1, 0.1)}
    lasso_reg = Lasso(max_iter=max_iter)
    grid_search = GridSearchCV(lasso_reg, param_grid, cv=cv, scoring=scoring, refit=True, n_jobs=-1)
    grid_search.fit(train_x, train_y)
    tmp_best_param = grid_search.best_params_['alpha']
    param_grid = {'alpha': np.arange(np.round(tmp_best_param - 0.09, 2), np.round(tmp_best_param + 0.1, 2), 0.01)}
    grid_search = GridSearchCV(lasso_reg, param_grid, cv=cv, scoring=scoring, refit=True, n_jobs=-1)
    grid_search.fit(train_x, train_y)
    # print('Lasso Regression Best Parameters: ', grid_search.best_params_)
    return grid_search.best_estimator_


def param_grid_search_elastic_net(train_x, train_y, max_iter=10000, cv=10, scoring='neg_mean_squared_error'):
    """Return the best elastic-net regression model."""
    param_grid = {'alpha': np.arange(0.1, 1, 0.1), 'l1_ratio': np.arange(0.1, 1.1, 0.1)}
    elastic_net = ElasticNet(max_iter=max_iter)
    grid_search = GridSearchCV(elastic_net, param_grid, cv=cv, scoring=scoring, refit=True, n_jobs=-1)
    grid_search.fit(train_x, train_y)
    tmp_best_param = grid_search.best_params_['alpha']
    param_grid = {'alpha': np.arange(np.round(tmp_best_param - 0.09, 2), np.round(tmp_best_param + 0.1, 2), 0.01),
                  'l1_ratio': np.arange(0.1, 1.1, 0.1)}
    grid_search = GridSearchCV(elastic_net, param_grid, cv=cv, scoring=scoring, refit=True, n_jobs=-1)
    grid_search.fit(train_x, train_y)
    # print('Elastic Net Regression Best Parameters: ', grid_search.best_params_)
    return grid_search.best_estimator_


def get_test_rmse(model, test_x, test_y):
    """Return the root mean squared error of the trained model based on the test dataset."""
    test_y_hat = model.predict(test_x)
    test_mse = mean_squared_error(test_y, test_y_hat)
    test_rmse = np.sqrt(test_mse)
    return test_rmse


def get_test_r2(model, test_x, test_y):
    """Return the r-squared of the trained model based on the test dataset."""
    test_y_hat = model.predict(test_x)
    test_r2 = r2_score(test_y, test_y_hat)
    return test_r2


# %%
dict_models_explore = {'ridge': Ridge(), 'lasso': Lasso(), 'elastic_net': ElasticNet(),
                       'svr': SVR(),
                       'decision_tree': DecisionTreeRegressor(),
                       'random_forest': RandomForestRegressor(),
                       'extra_trees': ExtraTreesRegressor(),
                       'gradient_boosting': GradientBoostingRegressor()}
dict_models = {'lasso': Lasso(), 'elastic_net': ElasticNet(),
               'random_forest': RandomForestRegressor(), 'gradient_boosting': GradientBoostingRegressor()}

# %%
train_set, test_set = train_test_split(df_LA_all_data_adj, test_size=0.2, random_state=42)
health_vars = ['ARTHRITIS', 'BPHIGH', 'CANCER', 'CASTHMA', 'CHD', 'COPD', 'DIABETES', 'HIGHCHOL', 'KIDNEY', 'MHLTH',
               'PHLTH', 'STROKE', 'TEETHLOST', 'BINGE', 'CSMOKING', 'LPA', 'OBESITY', 'SLEEP']
my_train_x = train_set.drop(columns=health_vars)  # drop creates a copy of the dataframe thus won't affect the original
my_test_x = test_set.drop(columns=health_vars)

# %%
# standardize features
scaler = StandardScaler()
my_train_x = pd.DataFrame(scaler.fit_transform(my_train_x.values), columns=my_train_x.columns)
my_test_x = pd.DataFrame(scaler.transform(my_test_x.values), columns=my_test_x.columns)
my_list_preds = []
my_dict_models = {}

for health_var in health_vars:
    print('\n-----------------------------------------------------')
    print('Outcome Variable: ', health_var)
    my_train_y = train_set[health_var].copy()
    my_test_y = test_set[health_var].copy()
    # fine-tune models
    best_model = param_grid_search_lasso(my_train_x, my_train_y)
    print('Lasso Regression cross-validation score: ', get_mean_cv_score(best_model, my_train_x, my_train_y))
    print('Lasso Regression Test RMSE: ', get_test_rmse(best_model, my_test_x, my_test_y))
    print('Lasso Regression Test R-Squared: ', get_test_r2(best_model, my_test_x, my_test_y))
    # get predicted values for all observations
    na_preds = best_model.predict(pd.concat([my_train_x, my_test_x]))
    ser_preds = pd.Series(na_preds, name='p' + health_var)
    my_list_preds.append(ser_preds)
    my_dict_models[health_var] = best_model

my_df_preds = pd.concat(my_list_preds, axis=1)

# %%
my_features = my_train_x.columns


def get_lasso_coef(my_health_var, dict_best_models):
    feature_coef = pd.Series(index=my_features, data=dict_best_models[my_health_var].coef_)
    feature_nonzero_coef = feature_coef[feature_coef != 0]
    feature_nonzero_coef_sorted = feature_nonzero_coef.sort_values()
    return feature_nonzero_coef_sorted


print(get_lasso_coef('STROKE', my_dict_models))

# %%
my_indices = train_set.index.append(test_set.index)
df_LA_map_actual = pd.concat([df_LA_tracts, df_LA_pop, df_LA_all_data_adj], axis=1).loc[my_indices, :]
df_LA_map_actual['tract'] = 's' + df_LA_map_actual['TractFIPS']
df_LA_map_actual.reset_index(drop=True, inplace=True)
df_LA_map = pd.concat([df_LA_map_actual, my_df_preds], axis=1)
df_LA_map.rename(columns={'Population2010': 'POP2010'}, inplace=True)
# %%
na_mean = scaler.mean_
na_sd = np.sqrt(scaler.var_)
for index, feature in enumerate(my_features):
    if feature in ['P_LOWWAGEr', 'P_LOWWAGEe', 'HHWRKJOBEQ', 'D_X_EXCLAO']:
        df_LA_map['m_' + feature[2:]] = na_mean[index]
        df_LA_map['s_' + feature[2:]] = na_sd[index]
    elif feature == 'WRKSPERJOB':
        df_LA_map['m' + 'WKSPERJOB'] = na_mean[index]
        df_LA_map['s' + 'WKSPERJOB'] = na_sd[index]
    else:
        df_LA_map['m' + feature] = na_mean[index]
        df_LA_map['s' + feature] = na_sd[index]
# %%
# save_csv(df_LA_map, 'LA_map_web')
# %%
# create a mapping between feature names and the names of their means and standard deviations
dict_name_mean_sd = {}
for index, feature in enumerate(my_features):
    dict_name_mean_sd[feature] = []
    if feature in ['P_LOWWAGEr', 'P_LOWWAGEe', 'HHWRKJOBEQ', 'D_X_EXCLAO']:
        dict_name_mean_sd[feature].append('m_' + feature[2:])
        dict_name_mean_sd[feature].append('s_' + feature[2:])
    elif feature == 'WRKSPERJOB':
        dict_name_mean_sd[feature].append('m' + 'WKSPERJOB')
        dict_name_mean_sd[feature].append('s' + 'WKSPERJOB')
    else:
        dict_name_mean_sd[feature].append('m' + feature)
        dict_name_mean_sd[feature].append('s' + feature)


# %%
# output formula
def get_lasso_formula(my_health_var, dict_best_models):
    feature_nonzero_coef_sorted = get_lasso_coef(my_health_var, dict_best_models)
    s_formula = ""
    for feature_name, coef in feature_nonzero_coef_sorted.iteritems():
        term = "({coef}) * (([{factor}] * [{feature}] - [{mean_feature}]) / [{sd_feature}]) + " \
            .format(coef=coef,
                    factor='factor_' + feature_name[:10],
                    feature=feature_name[:10],
                    mean_feature=dict_name_mean_sd[feature_name][0][:10],
                    sd_feature=dict_name_mean_sd[feature_name][1][:10])
        s_formula += term
    s_formula += str(dict_best_models[my_health_var].intercept_)
    return s_formula


# %%
print(get_lasso_formula('STROKE', my_dict_models))

# %%
ser_lasso_coef = get_lasso_coef('STROKE', my_dict_models)
df_lasso_coef = pd.DataFrame({'feature_name': ser_lasso_coef.index, 'coef': ser_lasso_coef.values})
save_csv(df_lasso_coef, "LA_lasso_coef_stroke")
