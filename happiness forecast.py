from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from catboost import Pool, CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import joblib
import pandas as pd
import numpy as np

x = pd.read_csv("C:/Users/成行/Desktop/dataset/happiness_train_clean.csv")
y =x['happiness']
del x['happiness']

kfold = KFold(n_splits=15, shuffle = True, random_state= 12)#n_splits：表示划分几等份;shuffle：在每次划分时，是否进行洗牌;random_state：随机种子数

#silent=1时，输出中间过程;iterations表示最大树数,默认500;depth代表树深，最大16，建议在1到10之间，默认6；l2_leaf_reg代表L2正则化参数，默认3
model = CatBoostRegressor(colsample_bylevel=0.1,thread_count=6,silent=True,iterations=800, 
                          depth=5, 
                          learning_rate=0.051, 
                          loss_function='RMSE',
                          l2_leaf_reg = 3)
mse = []
i=0
for train, test in kfold.split(x,y):
    X_train = x.iloc[train]
    y_train = y.iloc[train]
    X_test = x.iloc[test]
    y_test = y.iloc[test]
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    err = mean_squared_error(y_true=y_test,y_pred=y_pred)
    mse.append(err)
    print(err)
    joblib.dump(filename="cat"+str(i),value=model)#模型持久化
    i+=1
print("catboost",np.mean(mse),mse)

#################################xgboost###############################
model = XGBRegressor(base_score=0.5,
                     booster='gbtree',#gbtree 树模型做为基分类器（默认）gbliner 线性模型做为基分类器
                     colsample_bylevel=0.1,
                     colsample_bytree=0.971,#含义：训练每棵树时，使用的特征占全部特征的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。
                     gamma=0.11,     #惩罚项系数，指定节点分裂所需的最小损失函数下降值
                     learning_rate=0.069,#含义：学习率，控制每次迭代更新权重时的步长，默认0.3。调参：值越小，训练越慢。典型值为0.01-0.2。
                     max_delta_step=0,
                     max_depth=3,     #含义：树的深度，默认值为6，典型值3-10，调参：值越大，越容易过拟合；值越小，越容易欠拟合。
                     min_child_weight=1,#含义：默认值为1。调参：值越大，越容易欠拟合；值越小，越容易过拟合（值较大时，避免模型学习到局部的特殊样本）
                     missing=None,
                     n_estimators=499,#含义：总共迭代的次数，即决策树的个数
                     n_jobs=-1,
                     nthread=50,
                     objective='reg:linear',
                     random_state=0,
                     reg_alpha=0.1,
                     reg_lambda=1,
                     scale_pos_weight=1,#正样本的权重，在二分类任务中，当正负样本比例失衡时，设置正样本的权重，模型效果更好。例如，当正负样本比例为1:10时，scale_pos_weight=10。
                     seed=None,
                     silent=True,   #silent=1时，输出中间过程
                     subsample=1.0)  #含义：训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1。调参：防止overfitting。
mse = []
i = 0
for train, test in kfold.split(x,y):
    X_train = x.iloc[train]
    y_train = y.iloc[train]
    X_test = x.iloc[test]
    y_test = y.iloc[test]
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    xg_mse = mean_squared_error(y_true=y_test,y_pred=y_pred)
    mse.append(xg_mse)
    print("xgboost",xg_mse)
    joblib.dump(filename="xg"+str(i),value=model)
    i+=1
print("xgboost",np.mean(mse),mse)
########################gbdt################################################

model = GradientBoostingRegressor(alpha=0.9,
                                  criterion='friedman_mse',
                                  init=None,#影响了输出参数的起始化过程
                                  learning_rate=0.051,
                                  loss='ls',
                                  max_depth=4,#定义了树的最大深度
                                  max_features=10,#决定了用于分类的特征数，是人为随机定义的。
                                  max_leaf_nodes=None,#定义了决定树里最多能有多少个终点节点
                                  min_impurity_decrease=0.0,
                                  min_impurity_split=None,
                                  min_samples_leaf=1,#定义了树中终点节点所需要的最少的样本数
                                  min_samples_split=2,#定义了树中一个节点所需要用来分裂的最少样本数
                                  min_weight_fraction_leaf=0.0,#终点节点所需的样本数占总样本数的比值
                                  n_estimators=600,#定义了需要使用到的决定树的数量
                                  presort='auto',#决定是否对数据进行预排序，可以使得树分裂地更快
                                  random_state=3,#作为每次产生随机数的随机种子
                                  subsample=0.98,#训练每个决定树所用到的子样本占总样本的比例，而对于子样本的选择是随机的
                                  verbose=0,#决定建模完成后对输出的打印方式： 0：不输出任何结果（默认）1：打印特定区域的树的输出结果>1：打印所有结果
                                  warm_start=False)#使用它我们就可以用一个建好的模型来训练额外的决定树，能节省大量的时间，对于高阶应用我们应该多多探索这个选项。

mse = []
i = 0
for train, test in kfold.split(x,y):
    X_train = x.iloc[train]
    y_train = y.iloc[train]
    X_test = x.iloc[test]
    y_test = y.iloc[test]
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    gbdt_mse = mean_squared_error(y_true=y_test,y_pred=y_pred)
    mse.append(gbdt_mse)
    print("gbdt",gbdt_mse)
    joblib.dump(filename="gbdt"+str(i),value=model)
    i+=1
print("gbdt",np.mean(mse),mse)

#带权平均融合CatBoostRegressor + xgboost + gbdt现有模型计算：
from catboost import Pool, CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import pandas as pd

x = pd.read_csv("C:/Users/成行/Desktop/dataset/happiness_train_clean.csv")
y =x['happiness']
del x['happiness']

catmse = []
xgmse = []
gbdtmse = []
lrmse = []
i = 0
for train, test in kfold.split(x,y):
    X_train = x.iloc[train]
    y_train = y.iloc[train]
    X_test = x.iloc[test]
    y_test = y.iloc[test]

    cat=joblib.load("cat"+str(i)) 
    catX = cat.predict(X_test)
    cat_mse = mean_squared_error(y_true=y_test,y_pred=catX)
    print("\ncat mse:",cat_mse)
    catmse.append(cat_mse)
    
    xg=joblib.load("xg"+str(i))
    xgX = xg.predict(X_test)
    xg_mse = mean_squared_error(y_true=y_test,y_pred=xgX)
    print("xg mse:",xg_mse)
    xgmse.append(xg_mse)

    gbdt=joblib.load("gbdt"+str(i))
    gbdtX = gbdt.predict(X_test)
    gbdt_mse = mean_squared_error(y_true=y_test,y_pred=gbdtX)
    print("gbdt mse:",gbdt_mse)
    gbdtmse.append(gbdt_mse)
    
    res = np.c_[catX,xgX,gbdtX]#拼接成一个矩阵
    lr = Ridge(fit_intercept=False, alpha=75)#岭回归，alpha正则化强度; 必须是正浮点数。fit_intercept设置为false，则不会在计算中使用截距
    lr.fit(res,y_test)
    print(lr.coef_)
    y_pred = lr.predict(res)
    lr_mse = mean_squared_error(y_true=y_test,y_pred=y_pred)
    print("lr mse:",lr_mse)
    lrmse.append(lr_mse)
    
    joblib.dump(filename="lr"+str(i),value=lr)
    i+=1
    
print("\ncatmse:",np.mean(catmse))
print("xgmse:",np.mean(xgmse))
print("gbdtmse:",np.mean(gbdtmse))
print("lrmse:",np.mean(lrmse))

x_test_data=pd.read_csv("C:/Users/成行/Desktop/dataset/happiness_test_clean.csv")
d=pd.read_csv("C:/Users/成行/Desktop/dataset/happiness_submit.csv")
def cut(arr):
    arr2 = []
    for x in arr:
        if x<=1:
            arr2.append(1)
        elif x>=5:
            arr2.append(5)
        else :
            arr2.append(round(x))
    return arr2
prediction = []
pre=[]
i=0
while i<15:
    cat=joblib.load("cat"+str(i))
    xg=joblib.load("xg"+str(i))
    gbdt=joblib.load("gbdt"+str(i))
    
    catX = cat.predict(x_test_data)
    xgX = xg.predict(x_test_data)
    gbdtX = gbdt.predict(x_test_data)
    res = np.c_[catX,xgX,gbdtX]
    prediction.append(lr.predict(res))
    i=i+1
i=1
while i<15:
    prediction[0]=prediction[0]+prediction[i]
    i=i+1
print(prediction[0])
pre.append(prediction[0]/i)
pre[0]=cut(pre[0])
df = pd.DataFrame({'id':d.id, 'happniess': pre[0]})
df.to_csv('C:/Users/成行/Desktop/dataset/happiness_submit.csv', index=None)
print("done")


