# Learn the following skills by participating in the Tianchi Newcomer Competition:
1. Skilled in using jupyter notebook for basic data cleaning and processing.
2. Learn more advanced machine learning algorithms.
3. Skilled in using python related libraries. For example: pandas, numpy, scikit-learn, matplotlib, etc.
4. Master the common syntax using python programming. For example: round(), joblib.dump(), iloc(),oc() and so on.


# 机器学习应用开发的一般步骤（总结）<br>
## 数据采集并标记
    天池的比赛我们可以从网站直接获取得到数据集，这是非常方便的。然而很实际项目中，很多情况下需要我们自己去采集数据，并尽可能多的去收集特征并打上标签，虽然这是一个很繁琐很庞大的工作，但是一般情况下数据越多，特征越全，训练出来的模型才越准确。
## 数据清洗<br>
    在幸福感预测的数据清洗的过程中，我初步掌握了jupyternotebook的使用方法，并使用该工具进行训练数据的清洗，例如删除某个特征的异常值、填充某些特征的缺失值、进行特征之间的单位变换等操作。随后利用清洗后的数据进行了数据分析比如分年龄段和性别等特征来分析幸福指数、求各特征之间的相关性矩阵等。<br>
## 特征选择<br>
    我们数据集里的特征并不都是有用的，往往会存在一些冗余特征，这些特征非但对最终的结果作用很小，而且会降低模型的运算效率。我在幸福感预测这个模型中特征的选择是根据特征与幸福指数的相关性指标得到的（选取与幸福感相关性大的前20个特征），或者使用较多的还有PCA主成分分析算法。<br>
## 模型选择<br>
    模型的选择和问题领域、数据量大小、训练时长、模型的精确度等多方面有关。通过选择幸福感预测这个比赛的模型我学习到了机器学习很多的理论基础例如过拟合与欠拟合、成本函数、模型准确性、学习曲线等。<br>
## 模型训练和测试<br>
    在训练过程中我比较深入的掌握了交叉验证的概念。<br>
## 模型性能评估和优化<br>
    模型一旦确定后，模型的参数会影响模型性能。关于参数的调节，这也是门学问，需要丰富的经验。<br>
