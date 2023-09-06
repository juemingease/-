import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
# 绘图
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score  # 导入Silhouette Score

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
df = pd.read_excel(r'./Sheet12.xlsx', sheet_name=0)
df.fillna(0, inplace=True)
df = df.drop(columns=['文物编号','纹饰','颜色','表面风化'])

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
x = df[['二氧化硅(SiO2)','氧化钠(Na2O)','氧化钾(K2O)','氧化钙(CaO)','氧化镁(MgO)',\
        '氧化铝(Al2O3)','氧化铁(Fe2O3)','氧化铜(CuO)','氧化铅(PbO)'	,'氧化钡(BaO)'	,'五氧化二磷(P2O5)'	,'氧化锶(SrO)',	'氧化锡(SnO2)','二氧化硫(SO2)']]
y = df['类型']
# 将类别标签转换为字符串类型
y = y.astype(str)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
# 用决策树进行预测
dec = DecisionTreeClassifier()
dec.fit(x_train, y_train)
print("预测的准确率为：", dec.score(x_test, y_test))

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import pydotplus
from IPython.display import Image

# 导出决策树为Graphviz格式
export_graphviz(
    dec,
    out_file="tree.dot",
    feature_names=x.columns,
    class_names=y.unique(),
    filled=True,
    rounded=True,
    special_characters=True,
    fontname="SimHei",  # 添加此行，使用SimHei字体
)

# 将Graphviz文件转换为图形或查看器中的可视化结果
import pydotplus
from IPython.display import Image

dot_data = export_graphviz(dec, out_file=None, feature_names=x.columns, class_names=y.astype(str).unique(), filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

df1 = df[df["类型"]==1]
df2 = df[df["类型"]==2]
df1 = df1.drop(columns='类型')
df2 = df2.drop(columns='类型')
sample_label1 =df1['文物采样点'].values
sample_label2 =df2['文物采样点'].values
df1 = df1.drop(columns='文物采样点')
df2 = df2.drop(columns='文物采样点')

def Hierarchical_Clustering(data,sample_labels):
    # 层次聚类
    z = linkage(data, "ward", metric='euclidean')
    # 类间距离为最短距离，距离计算使用欧式距离
    print(z)  # 聚类过程

    # 画聚类图
    fig, ax = plt.subplots(figsize=(15, 9))  # 图片尺寸
    dendrogram(z ,labels=sample_labels, ax=ax, orientation='top')
    plt.title("层次聚类树状图")
    plt.xlabel("文物采样点")
    plt.ylabel("距离")
    plt.show()

    # 计算不同k值下的Silhouette Score
    silhouette_scores = []
    for k in range(2, 11):  # 从2到10尝试不同的k值
        cluster_labels = AgglomerativeClustering(n_clusters=k).fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        print(f"For n_clusters={k}, the Silhouette Score is {silhouette_avg}")

    # 绘制Silhouette Score的曲线
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--')
    plt.xlabel("聚类数量 (k)")
    plt.ylabel("轮廓系数")
    plt.title("不同聚类数量的轮廓系数")
    plt.grid(True)
    plt.show()

    # 敏感性分析
    for i in range(0,14):
        data1 = data.drop(data.columns[i], axis=1)
        # 层次聚类
        z = linkage(data1, "ward", metric='euclidean')
        # 类间距离为最短距离，距离计算使用欧式距离
        print(z)  # 聚类过程

        # 画聚类图
        fig,ax= plt.subplots(figsize=(15, 9))  # 图片尺寸
        dendrogram(z, labels=sample_labels, ax=ax, orientation='top')
        plt.title("层次聚类树状图")
        plt.xlabel("文物采样点")
        plt.ylabel("距离")
        plt.show()

        # 计算不同k值下的Silhouette Score
        silhouette_scores = []
        for k in range(2, 11):  # 从2到10尝试不同的k值
            cluster_labels = AgglomerativeClustering(n_clusters=k).fit_predict(data1)
            silhouette_avg = silhouette_score(data1, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            print(f"For n_clusters={k}, the Silhouette Score is {silhouette_avg}")

        # 绘制Silhouette Score的曲线
        plt.figure(figsize=(8, 6))
        plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='--')
        plt.xlabel("聚类数量 (k)")
        plt.ylabel("轮廓系数")
        plt.title("不同聚类数量的轮廓系数")
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    Hierarchical_Clustering(df1,sample_label1)
    Hierarchical_Clustering(df2,sample_label2)
