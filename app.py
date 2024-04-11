import pickle
import streamlit as st
import pandas as pd
import warnings

# Ignore all UserWarnings
warnings.filterwarnings('ignore', category=UserWarning)


loaded_model = pickle.load(open("model_carprice.sav", 'rb'))

def predict_car_price(mileage,age):
    data = pd.DataFrame({'Mileage': [mileage],'Age(yrs)': [age]})
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
        prediction = predict_car_price(mileage,age)
        st.write('Predicted Price: (in $) ',prediction)
if __name__ == '__main__':
    main()
