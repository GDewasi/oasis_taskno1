import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

# initializing the StandardScaler Object
scaler=StandardScaler()


# loading our model file (model.sav) into this program
model=pickle.load(open('model.sav','rb'))

# loading scaler object file in the program
scaler=pickle.load(open('scaler.sav','rb'))

# function for prediction
def flowerPrediction(s_length,s_width,p_length,p_width):

    input_data=np.array([[s_length,s_width,p_length,p_width]])
    
    # transforming the ndarray values
    input_data_scaler=scaler.transform(input_data)

    # prediction part
    pred=model.predict(input_data_scaler)

    if(pred==1):
        return "**Versicolor** 🌷"
    elif(pred==2):
        return "**Virginica** 🌷"
    else:
        return "**Setosa** 🌷"



# main() for web app interface and input tasks

def main():

    
    # for wide look 
    st.set_page_config(layout="wide")


    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.pexels.com/photos/7232493/pexels-photo-7232493.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    
    html_temp="""

    <div style="background-color:tomato;padding:10xp">
    <h2 style="color:white;text-align:center;">Iris Flower Predicition  Model</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)


    s_length = int( st.slider('**Sepal Length (in cm)**',4.3,7.9,5.0) )
    s_width = int( st.slider('**Sepal Width (in cm)**',2.2,4.0,3.0) )
    p_length = int( st.slider('**Petal Length (in cm)**',1.0,6.9,5.0) )
    p_width = int( st.slider('**Petal width (in cm)**',0.1,2.5,1.5) )

    
    # creating the object, for displaying the predicted string
    whichCategory= ''

    # button for prediction
    if st.button("Predict"):
        whichCategory=flowerPrediction(s_length,s_width,p_length,p_width)

    st.success(whichCategory)


if __name__ == '__main__':
    main()
