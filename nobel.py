
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


st.write(''' # Nobel Prize category prediction ''')
st.image("Nobel.png", caption="Su creador fue el inventor sueco Alfred Nobel mediante su testamento en 1895.")

st.header('Texto')

def user_input_features():

  texto = st.text_input("Enter The text to be evaluated")

  user_input_data = {'Text': texto}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

nobel =  pd.read_csv('df_novel.csv', encoding='latin-1')
X = nobel.Text
y = nobel.label

vect = CountVectorizer()
x_dtm = vect.fit_transform(X)

nb = MultinomialNB()
nb.fit(x_dtm, y)

df_dtm = vect.transform(df['Text'])
prediction = nb.predict(df_dtm)

#{'physics':0, 'medicine':1, 'peace':2, 'literature':3, 'chemistry':4, 'economics':5}
#'Physics', 'Medicine', 'Peace', 'Literature', 'Chemistry', 'Economics'
st.subheader('predicción')
if prediction == 0:
  st.write('physics')
elif prediction == 1:
  st.write('medicine')
elif prediction == 2:
  st.write('peace')
elif prediction == 3:
  st.write('literature')
elif prediction == 4:
  st.write('chemistry')
elif prediction == 5:
  st.write('economics')
else:
  st.write('whit out prediction')
