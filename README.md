# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas and matplotlib.pyplot

2. Read the dataset and transform it

3. Import KMeans and fit the data in the model

4. Plot the Cluster graph

 

## Program:

Program to implement the K Means Clustering for Customer Segmentation.

~~~
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\admin\Downloads\Mall_Customers.csv")
data.head()
data.info()
data.isnull().sum()

from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init = "k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("Elbow method")

km = KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])

y_pred = km.predict(data.iloc[:,3:])
y_pred

data["cluster"] = y_pred

df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster 0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster 1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster 2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster 3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster 4")

plt.legend()
plt.title("customer segmentation")
~~~
~~~
Developed by: GURUPARAN G
RegisterNumber: 212224220030 
~~~

## Output:

![Screenshot 2025-04-21 031124](https://github.com/user-attachments/assets/dca3ef37-918f-4638-bcd7-0e7a965a2b2c)

![Screenshot 2025-04-21 031150](https://github.com/user-attachments/assets/1a574fcb-a139-431a-83c1-fc3b88e27baf)

![Screenshot 2025-04-21 031211](https://github.com/user-attachments/assets/3b0147e4-0a57-4504-8798-07857de55137)

![Screenshot 2025-04-21 031347](https://github.com/user-attachments/assets/5333fd46-e16e-4968-af8f-639c962e67f8)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
