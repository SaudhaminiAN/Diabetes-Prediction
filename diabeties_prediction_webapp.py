import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model=pickle.load(open('C:/Users/saudh/Downloads/diabetes/trained_model (2).sav','rb'))

#creating a function for prediction
def diabetes_prediction(input_data):
    
    #changing the input_data to a numpy array
    input_data_as_numpy_array=np.asarray(input_data)

    #reshape the np array as we are predicting for one instance
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0])==0:
      return 'The person is not diabetes'
    else:
      return 'The person is diabetes'
  

def main():
    
    #giving a title
    st.title('Diabetes Prediction Web App')
    
    #getting input data from the user
    
    Pregnancies=st.text_input('Number of Pregnancies')
    Glucose=st.text_input('Value of Glucose')
    BloodPressure=st.text_input('Value of BloodPressure')
    SkinThickness=st.text_input('Value of SkinThickness')
    Insulin=st.text_input('Level of Insulin')
    BMI=st.text_input('Value of BMI')
    DiabetesPedigreeFunction=st.text_input('Value of DiabetesPedigreeFunction')
    Age=st.text_input('Age of the person')
    
    #code for prediction
    diagnosis=''
    
    #creating a button for prediction
    if st.button('Diabetes test results'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    st.success(diagnosis) 
    
   
    
   
if __name__ == '__main__':
    main()
    
    
    

    
    