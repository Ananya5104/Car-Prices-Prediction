import pickle
import streamlit as st
import pandas as pd
import warnings
import sklearn

# Ignore all UserWarnings
warnings.filterwarnings('ignore', category=UserWarning)


loaded_model = pickle.load(open("trained_model.sav", 'rb'))

def predict_car_price(year,mileage):
    data = pd.DataFrame({'Prod. year': [year],'Mileage': [mileage]})
    # input_data = (mileage,age)
    # input_data_as_numpy_array = np.asarray(input_data)
    # input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(data)
    return prediction

def main():
    st.title("Car Price Prediction (ReSale)")
    age = st.text_input('Age')
    mileage = st.text_input('Mileage')
    prediction = ''
    if st.button('Predict'):
        prediction = predict_car_price(age,mileage)
        st.write('Predicted Price: (in $) ',prediction)
if __name__ == '__main__':
    main()
