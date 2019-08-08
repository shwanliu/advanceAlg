# Task 1

## 随机森林算法梳理
1. 集成学习的概念 \
    集成学习（ensemble learning）是指通过构建并组合多个学习期来完成学习任务，集成学习将多个学习器进行结合，可以获得比个体的学习器更加显著的泛化能力，通过[随机有放回采样]方式获得多个数据子集进行多个学习器的训练，

2. 个体学习器的概念 \
     个体学习器通常是由单一的算法在一组数据上学习得到的，也被称为基学习器，一般在bagging和boosting中使用得到，一般为决策树（ID3、C4.5、CART）等不稳定的学习器

3. boosting 和 bagging的概念、异同点 
    * boosting的概念：boosting的基学习器采用按序进行训练，所有基学习器的训练数据都是一样的，最主要的改变是每训练出一个基学习器，就要相应的修改每一个样本在该数据中权重，以及赋予每一个基学习器的可信权重，是一个加法模型得到的强学习器，最后结合所有基学习器的结果

    * bagging的概念：bagging是对所有数据进行随机有放回的抽样出多个子训练集，每一个子训练集进行对应的基学习器的训练，得到多个基学习器，最后结合每一个基学习器的结果进行预测

    * 异同点： \
    a) 采用的训练数据不同，bagging是在原始数据集中有放回的随机抽样，每一个基学习器的权重相等；而boosting的训练数据是一样的，但每个样本在学习器中的权重会随着上一个学习器的学习结果发生变化 \
    b) bagging可以并行训练；boosting只能按序，前向分步算法 \
    c) bagging 中整体模型的期望近似于基模型的期望，所以整体模型的偏差相似于基模型的偏差，因此bagging中的基模型为强模型（强模型有低偏差高方差）；而在boosting中的基模型为弱模型，弱模型的方差会比较大

4. 不同的结合策略（平均法、投票法、学习法）
    * 平均法：直接平均，加权平均 \
        使用范围：规模大的集成模型，学习的权重比较多，加权平均法容易导致过拟合 \
        个体学习器性能相差较大时使用加权平均法，相近用直接平均法

    * 投票法：\
        a) 绝对多数投票法 \
        b) 相对多数投票法：如果有多组最高票数一样的，随机选其中一个 \
        c) 加权投票法：预测分数越高，其投票的权重越大 

    * 学习法：
        a) Stacking描述：通过另一个学习器将若学习器的结果来进行结合 \
        Stacking的思想是一种有层次的融合模型，比如我们将用不同特征训练出来的三个GBDT模型进行融合时，我们会将三个GBDT作为基层模型，在其上在训练一个次学习器（通常为线性模型LR）,用于组织利用基学习器的答案，也就是将基层模型的答案作为输入，让次学习器学习组织给基层模型的答案分配权重。

        b)Blending：和stacking方式很类似，相比Stcking更简单点 \ 
        两者区别是：blending是直接准备好一部分10%留出集只在留出集上继续预测，用不相交的数据训练不同的 Base Model，将它们的输出取（加权）平均。实现简单，但对训练数据利用少了。          


5. 随机森林的思想 \
    用随机的方式建立多棵树，组成一个森林，森林由很多的决策树组成，随机森林的每一棵决策树之间是每一棵树之间是没有关联的。首先，使用bootstrap方法生成m个训练集，对于每个训练集，构造一颗决策树，在构建该树的时找特征进行分裂的时候，并不是对多所有特征找到能使得最好的特征分裂点，而是在特征中也进行随机选抽取一部分特征，在抽到的特征中间找到最优解，应用于节点，进行分类，随机森林实际上对样本和特征都进行了随机采样，所以避免了过拟合的发生。

6. 随机森林的推广 \
    extra trees是RF的一个变种，原理几乎和RF一摸一样，仅有区别有：
    1) 对于每个决策树的训练机，RF采用的实际随机采样bootstrap来选择采样集作为每个决策树的训练集，而extra trees一般不采用随机采样，即每个决策树采用原始的训练集。
    2) 在选定划分特征后，RF的决策树会基于信息增益，基尼系数，信息增益比等的原则，选取一个最优特征值划分点，这和传统的决策树相同，但是extra trees是之际从所有的特征中选随机选择一个特征值来划分决策树 \
    从2)可以看出，由于随机选择了特征值的划分点位，而不是最优点位，这样会导致生成的决策树的规模一般会大于RF所生成的决策树，即模型的方差相对于RF进一步减少，但是bias相对于RF进一步增大，在某些时候，extra trees的方差相对于RF进一步减少，但是bias相对于RF进一步增大，在某些时候，extra trees的泛华能力比RF更好

7. 随机森林的优缺点 \
优点：\
a) 随机森林能解决分类和回归两种类型的问题，表现良好，由于是集成学习，方差和偏差都比较低，泛化性能优越；\
b) 随机森林对于高维数据集的处理能力很好，他可以处理成千上万的输入变量，并确定最重要的变量，一次被认为是一个不错的降维方法。此外，该模型能够输出特征的重要程度 \
c) 对于缺失值不会敏感； \
d) 随机森林可以解决数据不平衡的问题 \
e) 支持并发

8. 随机森林在sklearn中的参数解释 \
    sklear.ensemble 模块包含两个基于随机决策树的平均算法：RandomForest算法和Extra-Tree算法，前面已经进行了简单的介绍了，这两种算法都是专门为树而设计的扰动和组合技术（perturb-and-combine techniques）。\
    这中技术通过在分类器构造过程中引入随机性来创建一组不同的分类器，集成分类器的预测结果就是单个分类器预测结果的平均值 \
    与其它分类器一样，森林分类器必须拟合（fit）两个数组：保存训练样本的数组（或稀疏或稠密的）X，大小为[n_samples,n_features]，以及保存训练样本的目标值（类标签）的数组，大小为[n_samples]：
    > from sklearn.ensemble import RandomForestClassifier \
    > X = [[0, 0], [1, 1]] \
    > Y = [0, 1] \
    > clf = RandomForestClassifier(n_estimators=10) \
    > clf = clf.fit(X, Y) \
    sklear中对于分类问题使用RandomForestClassifier ，对于回归问题使用RandomForestRegressor

    RF参数：
    1. n_estimators：决策树的个数，太容易欠拟合没太大不能显著提升模型
    2. criterion：选择最优特征分裂点的评价标准
    3. max_depth：决策树的最大深度，等于None则不限制，样本过多、特征多的情况下，建议决策树的的深度进行限制
    4. min_samples_split：内部节点再划分所需最小样本数，到达该节点，如果大于该参数就要继续进行分裂
    5. min_samples_leaf：叶子节点最少样本数。若叶子节点样本数小于min_samples_leaf，则对该叶子节点和兄弟节点进行剪枝，只留下该叶子节点的父节点。
    6. min_weight_fraction_leaf：叶子节点最小的样本权重和
    7. max_features：寻找最优分裂点时候的特征数量
    8. max_leaf_nodes：最大叶子节点数，可以防止过拟合，默认是"None”，即不限制最大的叶子节点数。
    9. min_impurity_decrease ：A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
   10. min_impurity_split： 这个值限制了决策树的增长，如果某节点的不纯度(基尼系数，信息增益，均方差，绝对差)小于这个阈值，则该节点不再生成子节点。即为叶子节点 。
   11. bootstrap：对样本集进行有放回抽样来构建树
   12. oob_score：采用袋外样本来评估模型的好坏
   13. n_jobs：The number of jobs to run in parallel for both fit and predict
   14. random_state
   15. verbose
   16. warm_start
   17. class_weight：指定样本各类别的的权重，主要是为了防止训练集某些类别的样本过多，导致训练的决策树过于偏向这些类别。

    RF属性：
    1. estimators_： 所有决策树基分类器 list
    2. classes_：The classes labels (single output problem), or a list of arrays of class labels (multi-output problem).
    3. n_classes_：类别的数量
    4. n_features_：特征的数量，在拟合的时候
    5. n_output_：The number of outputs when fit is performed.
    6. feature_importance_：Return the feature importances (the higher, the more important the feature).
    7. oob_score_：Score of the training dataset obtained using an out-of-bag estimate.
    8. oob_decision_function_：


    参考：https://sklearn.apachecn.org/docs/0.21.3/12.html
9. 随机森林的应用场景：\
* 数据维度相对低（几十维），同时对准确性有比较高的要求
* 不需要很多参数的调整
* 训练速度快
* 可以处理缺省值
* 由于有袋外数据（OOB），可以在模型生成过程中取得真实误差的无偏估计，且不损失训练数据量
* 在训练过程中，能够检测到feature间的互相影响，且可以得出feature的重要性，具有一定参考意义，由于每棵树可以独立、同时生成、容易做成并行化
*由于实现简单、精度高、抗过拟合能力强、当面对非线性数据时，适和做base model

参考资料: \
    西瓜书 \
    cs229吴恩达机器学习课程 \
    李航统计学习 \
    谷歌搜索 \
    公式推导参考：http://t.cn/EJ4F9Q0