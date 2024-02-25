from pyearth import Earth
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# # Reading the Cleaned Data
#
# data_=pd.read_csv("ELS_dis_removed.csv")
# # data.drop(['sex','region_first','sander_index'], inplace=True, axis=1)
# # data=data_.dropna()
# data.pop(data.columns[0])
# #
data=pd.read_csv("crime.csv")
data.drop(['V6', 'V7', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
       'V20', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30',
       'V31', 'V32', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41',
       'V42', 'V43', 'V44', 'V45', 'V47', 'V48', 'V53', 'V54', 'V57', 'V58',
       'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68',
       'V69', 'V70', 'V71', 'V72', 'V73', 'V75', 'V76', 'V78', 'V79', 'V80',
       'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89', 'V90',
       'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100',
       'V101', 'V119', 'V120', 'V121', 'V126'], inplace=True, axis=1)
data=data.rename(columns={
       'y':'violent_crime_rates'})
# # # #Split & Scale:
# # #
# x=data.drop(columns={'F3_GPA(all)'}).copy()
# y=data.loc[:,'F3_GPA(all)'].copy()
# #
# #
x=data.drop(columns={'violent_crime_rates'}).copy()
y=data.loc[:,'violent_crime_rates'].copy()
# #
# print(x.columns)
# print(x.columns.get_loc('Hispanic'))
# # print(x.columns[36])
# #
# x=x.to_numpy()
# y=y.to_numpy()

#
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=456)


# # #scaler = MinMaxScaler()
# # #scaler = StandardScaler()
# # # X_train_scaled = scaler.fit_transform(X_train)
# # # X_test_scaled = scaler.transform(X_test)
# # # model
# #
# # model = Earth(max_degree=1)
# # model.fit(x, y, disparity_indices=[0],petha=0.8,fair_coef=True) #0.09, 0.01, 0.001
# # print(model.summary())
# #
#
# # df=data.dropna()
# # x=df.drop(columns={'violent_crime_rates'}).copy()
# # y=df.loc[:,'violent_crime_rates'].copy()
#
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=11)
#
# X_train_=X_train.to_numpy()
# y_train_=y_train.to_numpy()
#
# model = Earth(max_degree=1)
# # model = Earth(max_degree=1,enable_pruning=False)
#
# #Crime dataset
# model = model.fit(X_train_, y_train_, disparity_indices=[36,37,38,39,40],petha=0,fair_coef=False)
# print(model.summary())


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error



#*******************************************************ELS dataset****************************************************
# data=pd.read_csv("new2.csv")
# data.drop(['Unnamed: 0','gender_Male','Race_Amer. Indian/Alaska Native'], inplace=True, axis=1)
# #data.drop(['Unnamed: 0','gender_Male','Race_Amer. Indian/Alaska Native','F3_GPA(first year)','F3_Highest level of education','credits(first year)','F2_transferred','F2_grant'], inplace=True, axis=1)
# data=data.rename(columns={
#        'Race_Asian, Hawaii/Pac. Islander':'Asian', 'Race_Black or African American':'Black',
#        'Race_Hispanic':'Hispanic', 'Race_More than one race':'More_than_one', 'Race_White':'White'
# })


#
# df=data.dropna()
# import random
# random.seed(1000)
# seeds=[11,20,31,42]
# print('seeds',seeds)
# j=1
# petha = [0, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
# avg_mse_dif_asian = []
# avg_mse_dif_black = []
# avg_mse_dif_hispanic = []
# avg_mse_dif_MR = []
# avg_mse_dif_white = []
# avg_group_error = []
#
# for i in seeds:
#     print('seed is',i)
#     ## train/test split here
#     print('***** Running for Split'+str(j)+'*********\n')
#     from sklearn.model_selection import train_test_split
#     # from sklearn.preprocessing import MinMaxScaler
#
#     x=df.drop(columns={'F3_GPA(all)'}).copy()
#     y=df.loc[:,'F3_GPA(all)'].copy()
#     X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=i)
#     X_train_=X_train.to_numpy()
#     y_train_=y_train.to_numpy()
#     #*******************************************************************************
#     x.columns.get_loc('Asian')
#     mse_train = []
#     rsq_train = []
#     rsq_test = []
#     mse_test = []
#     mse_dif_asian = []
#     mse_dif_black = []
#     mse_dif_hispanic = []
#     mse_dif_MR = []
#     mse_dif_white = []
#     mse_diff_totals = []
#     mse_diff_seed = []
#     group_error = []
#     # run a loop for each petha of the list
#
#     for q in petha:
#         model = Earth(max_degree=1,enable_pruning=False)
#         ##Hadis dataset
#         model = model.fit(X_train_, y_train_, disparity_indices=[30,31,32,33,34],petha=q)
#     #***************************************************************Calculating mse_train, rsq_train***************************************************************
#         mse_ = model.mse_
#         rsq_ = model.rsq_
#         mse_train.append(mse_)
#         rsq_train.append(rsq_)
#
#         y_hat = model.predict(X_train)
#         # y_hat = model.predict(X_test)
#
#         # df_x_test = pd.DataFrame(X_test, columns=X_train.columns)
#         df_x_test = pd.DataFrame(X_train, columns=X_train.columns)
#         df_x_test.index=y_train.index
#         # df_x_test.index=y_test.index
#         df_yyhat = pd.concat([df_x_test, y_train.rename('y_test_actual')], axis=1)
#         # df_yyhat = pd.concat([df_x_test, y_test.rename('y_test_actual')], axis=1)
#         df_yyhat['y_test_predicted'] = y_hat
#
#         df_yyhat.loc[df_yyhat['Hispanic']==1, 'Race'] = 'Hispanic'
#         df_yyhat.loc[df_yyhat['Asian']==1, 'Race'] = 'Asian'
#         df_yyhat.loc[df_yyhat['Black']==1, 'Race'] = 'Black'
#         df_yyhat.loc[df_yyhat['More_than_one']==1, 'Race'] = 'MR'
#         df_yyhat.loc[df_yyhat['White']==1, 'Race'] = 'White'
#     #***************************************************************Calculating mse_test, rsq_test***************************************************************
#         from sklearn.metrics import r2_score
#         rsq_test_ = r2_score(df_yyhat['y_test_actual'], df_yyhat['y_test_predicted'])
#         rsq_test.append(rsq_test_)
#
#         from sklearn.metrics import mean_squared_error
#         mse_test_ = mean_squared_error(df_yyhat['y_test_actual'], df_yyhat['y_test_predicted'])
#         mse_test.append(mse_test_)
#     #***************************************************************Calculating mse_subgroup_diff_test***************************************************************
#
#         df_yyhat['MSE'] = np.power((df_yyhat['y_test_actual'] - df_yyhat['y_test_predicted']),2)
#         mse_asian = df_yyhat.loc[df_yyhat['Race'] == 'Asian', 'MSE'].mean()
#         mse_black = df_yyhat.loc[df_yyhat['Race'] == 'Black', 'MSE'].mean()
#         mse_hispanic = df_yyhat.loc[df_yyhat['Race'] == 'Hispanic', 'MSE'].mean()
#         mse_MR = df_yyhat.loc[df_yyhat['Race'] == 'MR', 'MSE'].mean()
#         mse_white = df_yyhat.loc[df_yyhat['Race'] == 'White', 'MSE'].mean()
#
#         mse_non_asian = df_yyhat.loc[df_yyhat['Race'] != 'Asian', 'MSE'].mean()
#         mse_non_black = df_yyhat.loc[df_yyhat['Race'] != 'Black', 'MSE'].mean()
#         mse_non_hispanic = df_yyhat.loc[df_yyhat['Race'] != 'Hispanic', 'MSE'].mean()
#         mse_non_MR = df_yyhat.loc[df_yyhat['Race'] != 'MR', 'MSE'].mean()
#         mse_non_white = df_yyhat.loc[df_yyhat['Race'] != 'White', 'MSE'].mean()
#
#         mse_dif_asian_ = abs(mse_asian - mse_non_asian)
#         mse_dif_black_ = abs(mse_black - mse_non_black)
#         mse_dif_hispanic_ = abs(mse_hispanic - mse_non_hispanic)
#         mse_dif_MR_ = abs(mse_MR - mse_non_MR)
#         mse_dif_white_ = abs(mse_white - mse_non_white)
#         group_error_ = (mse_dif_asian_ + mse_dif_black_ + mse_dif_hispanic_ + mse_dif_MR_ + mse_dif_white_)/5
#         # print('group_error_',group_error_)
#
#
#         mse_dif_asian.append(mse_dif_asian_)
#         mse_dif_black.append(mse_dif_black_)
#         mse_dif_hispanic.append(mse_dif_hispanic_)
#         mse_dif_MR.append(mse_dif_MR_)
#         mse_dif_white.append(mse_dif_white_)
#         group_error.append(group_error_)
#         # print('group_error',group_error)
#
#
#     avg_mse_dif_asian.append(mse_dif_asian)
#     avg_mse_dif_black.append(mse_dif_black)
#     avg_mse_dif_hispanic.append(mse_dif_hispanic)
#     avg_mse_dif_MR.append(mse_dif_MR)
#     avg_mse_dif_white.append(mse_dif_white)
#     avg_group_error.append(group_error)
#
#     # mse_diff_totals.append(np.mean(mse_diff_seed))
#     # print('mse_diff_totals',mse_diff_totals)
#
#     # fair_MARS_all_seeds=pd.DataFrame([avg_group_error, avg_mse_dif_asian,avg_mse_dif_black,avg_mse_dif_hispanic,avg_mse_dif_MR,avg_mse_dif_white])
#     # fair_MARS_all_seeds.index= ['avg_group_error','avg_Asian-Non_Asian', 'avg_Black-Non_Black', 'avg_Hispanic-Non_Hispanic','avg_MR-Non_MR','avg_White-Non_White']
#     # fair_MARS_all_seeds
#
#
#     # fair_MARS=pd.DataFrame([group_error,mse_train,rsq_train, mse_test, rsq_test, mse_dif_asian,mse_dif_black,mse_dif_hispanic,mse_dif_MR,mse_dif_white])
#
#     # fair_MARS.columns= petha
#     # fair_MARS.index= ['group_error','mse_train', 'rsq_train','mse_test','rsq_test', 'Asian-Non_Asian', 'Black-Non_Black', 'Hispanic-Non_Hispanic','MR-Non_MR','White-Non_White']
#     # fair_MARS
#     j= j+1

#*******************************************************Crime dataset****************************************************

petha = [0.01]
mse_train = []
rsq_train = []
rsq_test = []
mse_test = []
mae_test = []
mse_dif_black = []
group_error = []

# run a loop for each petha of the list
for i in petha:
    model = Earth(max_degree=1)

    #model = Earth(max_degree=1,enable_pruning=False)

    #Crime dataset
    X_train_=X_train.to_numpy()
    y_train_=y_train.to_numpy()
    model = model.fit(X_train_, y_train_, disparity_indices=[0],petha=i)

#***************************************************************Calculating mse_train, rsq_train***************************************************************
    mse_ = model.mse_
    rsq_ = model.rsq_
    mse_train.append(mse_)
    rsq_train.append(rsq_)

    y_hat = model.predict(X_test)

    df_x_test = pd.DataFrame(X_test, columns=X_test.columns)

    df_x_test.index = y_test.index

    df_yyhat = pd.concat([df_x_test, y_test.rename('y_test_actual')], axis=1)

    df_yyhat['y_test_predicted'] = y_hat
    df_yyhat.loc[df_yyhat['race']==1, 'Race'] = 'Black'
    df_yyhat.loc[df_yyhat['race']==0, 'Race'] = 'Others'

#***************************************************************Calculating MAE***************************************************************
    def mae(y_test_actual, y_test_predicted):
      y_test_actual, y_test_predicted = np.array(y_test_actual), np.array(y_test_predicted)
      return np.mean(np.abs(y_test_actual - y_test_predicted))

    mae_test_ = mae(df_yyhat['y_test_actual'], df_yyhat['y_test_predicted'])
    mae_test.append(mae_test_)
#***************************************************************Calculating mse_test, rsq_test***************************************************************
    from sklearn.metrics import r2_score
    rsq_test_ = r2_score(df_yyhat['y_test_actual'], df_yyhat['y_test_predicted'])
    rsq_test.append(rsq_test_)

    from sklearn.metrics import mean_squared_error
    mse_test_ = mean_squared_error(df_yyhat['y_test_actual'], df_yyhat['y_test_predicted'])
    mse_test.append(mse_test_)
#***************************************************************Calculating mse_subgroup_diff_test***************************************************************

    df_yyhat['MSE'] = np.power((df_yyhat['y_test_actual'] - df_yyhat['y_test_predicted']),2)
    mse_black = df_yyhat.loc[df_yyhat['Race'] == 'Black', 'MSE'].mean()
    mse_others = df_yyhat.loc[df_yyhat['Race'] != 'Black', 'MSE'].mean()
    mse_dif_black_ =  abs(mse_others - mse_black)
    mse_dif_black.append(mse_dif_black_)
    group_error_ = (mse_others + mse_dif_black_) / 5
    group_error.append(group_error_)

# fair_MARS=pd.DataFrame([mae_test, mse_train,rsq_train, mse_test, rsq_test,mse_dif_black])
#
# fair_MARS.columns= petha
# fair_MARS.index= ['mae_test','mse_train', 'rsq_train','mse_test','rsq_test', 'Others-Black']
# fair_MARS
# #print(model.trace())
# #MARS = model.forward_pass(X, y)
# #print(model.forward_trace())
print(model.summary())
#
# print("avg_group_error",group_error)



