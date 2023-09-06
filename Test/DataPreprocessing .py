# 数据读取
import pandas as pd
import matplotlib.pyplot as plt
data =pd.read_excel(r'./data.xlsx', sheet_name=0)
print(data)
# 将数据以不同数字代替
replace_dict1 = {"A": 1, "B": 2, "C": 3}
replace_dict2 = {"高钾":1,"铅钡":2}
replace_dict4 = {"无风化":0,"风化":1}
replace_dict3 = {
    "浅蓝": 1,
    "蓝绿": 2,
    "浅绿":3,
    "绿": 4,
    "深绿": 5,
    "深蓝": 6,
    "紫": 7,
    "黑": 8
}
data['纹饰'] = data['纹饰'].replace(replace_dict1)
data['类型'] = data['类型'].replace(replace_dict2)
data['颜色'] = data['颜色'].replace(replace_dict3)
data['表面风化'] = data['表面风化'].replace(replace_dict4)
print(data)
# 进行规律统计
condition = (data["纹饰"] == 1) & (data["类型"] == 2) & (data["表面风化"] == 1)
filtered_data = data[condition]

color_distribution = filtered_data["颜色"].value_counts()
# 绘制饼状图
plt.figure(figsize=(8, 8))
plt.pie(color_distribution, labels=color_distribution.index, autopct='%1.1f%%', startangle=140)
plt.title("Color Distribution")
plt.axis('equal')  # 使饼图为正圆
plt.show()
# 利用众数插补
data.iloc[18,3] = 1
data.iloc[47,3] = 4
# 进行规律统计
condition = (data["纹饰"] == 3) & (data["类型"] == 2) & (data["表面风化"] == 1)
filtered_data = data[condition]

color_distribution = filtered_data["颜色"].value_counts()
# 绘制饼状图
plt.figure(figsize=(8, 8))
plt.pie(color_distribution, labels=color_distribution.index, autopct='%1.1f%%', startangle=140)
plt.title("Color Distribution")
plt.axis('equal')  # 使饼图为正圆
plt.show()
# 利用众数插补
data.iloc[57,3] = 1
data.iloc[39,3] = 4
print(data)

data.to_excel(excel_writer='Sheet1.xlsx', sheet_name='sheet_1')