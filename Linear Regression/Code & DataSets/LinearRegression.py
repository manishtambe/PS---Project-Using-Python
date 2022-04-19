import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("Salary_Data.csv")

col_name = df.columns[0]
df = df.rename(columns={col_name: 'Years'})
col_name = df.columns[1]
df = df.rename(columns={col_name: 'Salary'})

reg = linear_model.LinearRegression()
reg.fit(df[['Years']], df.Salary)

plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.scatter(df.Years, df.Salary, color='red', marker='+')
plt.plot(df.Years, reg.predict(df[['Years']]), color='blue')
plt.show()

# print(reg.predict([[1.7]]))
# print(reg.coef_)
# print(reg.intercept_)
# print(reg.coef_*1.7+reg.intercept_)

d = pd.read_csv("Years.csv")
p = reg.predict(d)
d['Salary'] = p
d.to_csv('Predication.csv', index=False)
