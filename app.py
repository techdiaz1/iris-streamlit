import streamlit as st  
import numpy as np
from sklearn.datasets import load_iris                 #import the iris dataset 
from sklearn.neighbors import KNeighborsClassifier     #import the KNN algorithm 
st.title("IRIS FLOWER CLASSIFICATION")
var = load_iris() #load the dataset 

# divide the dataset into inputs and output 
x = var.data    #input 
y = var.target  #output

# call the knn classifier 
model = KNeighborsClassifier(n_neighbors=15)

# fit the model 
model.fit(x,y)

#to take inputs for the prediction we are creating sliders 
xmin = np.min(x,axis = 0)
xmax = np.max(x,axis = 1)

#Sliders creation 
sepal_length = st.slider("Sepal Length", float(xmin[0]),float(xmax[0]))
sepal_width  = st.slider("Sepal Width",  float(xmin[1]),float(xmax[1]))
petal_length = st.slider("Petal Length", float(xmin[2]),float(xmax[2]))
# petal_width  = st.slider("Petal Width",  float(xmin[3]),float(xmax[3]))

petal_width = st.number_input("Petal width",float(xmin[3]),float(xmax[3]))
#Predict the model 
y_pred = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])

#print the output class 
op = ['Iris-setosa','Iris-versicolor','Iris-virginica']
st.title(op[y_pred[0]])   #to get the output in single dimension we use y_pred[0] 
