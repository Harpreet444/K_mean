from sklearn.datasets  import load_iris
from matplotlib import pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="K_mean Clustering",page_icon='🧩',layout='wide')

data_set = load_iris()
df = pd.DataFrame(data_set.data,columns=data_set.feature_names)
df = pd.concat([df.drop(['sepal length (cm)','sepal width (cm)'],axis='columns'),pd.DataFrame(data_set.target,columns=['target'])],axis='columns')

st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: green;'>K-Mean Clustering</h1>", unsafe_allow_html=True)

st.write('''Exercise:
1. Use iris flower dataset from sklearn library and try to form clusters of flowers using petal width and length features. Drop other two features for simplicity.
2. Figure out if any preprocessing such as scaling would help here
3. Draw elbow plot and from that figure out optimal value of k''')

col1,col2 = st.columns(2)

col1.markdown("<h2 style='text-align: center; color: green;'>Scatter Graph</h2>", unsafe_allow_html=True)

df0 = df[df.target == 0]
df1 = df[df.target == 1]
df2 = df[df.target == 2]

fig, ax = plt.subplots()

ax.scatter(df0['petal length (cm)'], df0['petal width (cm)'], color='g', marker='*', label=data_set.target_names[0])
ax.scatter(df1['petal length (cm)'], df1['petal width (cm)'], color='r', marker='+', label=data_set.target_names[1])
ax.scatter(df2['petal length (cm)'], df2['petal width (cm)'], color='b', marker='o', label=data_set.target_names[2])

ax.set_xlabel('petal length (cm)')
ax.set_ylabel('petal width (cm)')
ax.legend()
ax.set_facecolor('#E6FFD7')

col1.pyplot(fig)


col2.markdown("<h2 style='text-align: center; color: green;'>Elbow plot</h2>", unsafe_allow_html=True)

sse = np.array([650.8953333333334,
 112.11808674985146,
 33.31546138455383,
 23.174013204508856,
 15.180391120763783,
 11.96398211437685,
 10.406647375079062,
 9.312651404151405])

# Create a figure and axis
fig, ax = plt.subplots()
ax.plot(range(1, 9), sse)

ax.set_xlabel('K')
ax.set_ylabel('Sum of square error')
ax.set_facecolor('#E6FFD7')
col2.pyplot(fig)

st.write("a) From elbow plot we can see the optimal value for k is 3")
st.write("b) Scaling the values dont make any improvement as values are already smaller and near to each other")

model = joblib.load("model.job")

co1,co2 = st.columns(2)

co1.markdown("<h2 style='text-align: center; color: green;'>Model</h2>", unsafe_allow_html=True)
length = co1.slider(label="petal length (cm)",min_value=1.0,max_value=7.0,step=0.1)
width = co1.slider(label="petal width (cm)",min_value=0.1,max_value=3.0,step=0.1)

btn = co1.button(label="Predict")

if btn:
    co1.code(data_set.target_names[model.predict([[length,width]])])

co2.markdown("<h2 style='text-align: center; color: green;'>Dataset structure</h2>", unsafe_allow_html=True)
co2.table(df.head())
