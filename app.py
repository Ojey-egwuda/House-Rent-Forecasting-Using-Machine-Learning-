import streamlit as st
import pandas as pd
import joblib
import time
import xgboost as xgb

# Load the pre-trained XGBoost model
xgb_model = joblib.load('best_xgboost_model.pkl')

# Streamlit setup
st.set_page_config(
    layout='wide',
    page_title='House Rent Prediction'
)

st.markdown("<h1 style='text-align: center; color: black;'> House Rent Prediction for Lagos State Nigeria Using Machine Learning </h1>", unsafe_allow_html=True)
# Coordinates of Lagos
# Coordinates of Lagos
lagos_coordinates = {'LAT': 6.465422, 'LON': 3.406448}

# Create a DataFrame with the coordinates
lagos_df = pd.DataFrame([lagos_coordinates])

# Display the map of Lagos
st.map(lagos_df)

# Developer's contact information
st.markdown("<h2 style='color: black;'>Contact the Developer</h2>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 16px;'>Connect with Egwuda Ojonugwa on : "
            "<a href='https://www.linkedin.com/in/egwudaojonugwa/' style='color: #00CED1;'>LinkedIn</a></p>",
            unsafe_allow_html=True)


# Sidebar for user input
st.sidebar.title("House Rent Estimator")

# House location selection
Location = st.sidebar.selectbox('House Location:', [
    'Lekki', 'Ikeja', 'Ipaja', 'Ojodu', 'Surulere', 'Onike-Yaba', 'Ketu', 'Phase1-Lekki', 'Shomolu',
    'Amuwo-Odofin', 'Ikotun-Igando', 'Ajah', 'Sangotedo', 'Ojota', 'Ogba', 'Ogudu', 'Phase1-Magodo',
    'Ikorodu', 'Alagomeji-Yaba', 'Victoria-Island', 'Osapa-London-Lekki', 'Ago-Palace-Okota', 'Medina-Gbagada',
    'Yaba', 'Ikoyi', 'Gbagada', 'Egbeda', 'Soluyi-Gbagada', 'Chevron-Lekki', 'Phase2-Magodo', 'Phase2-Gbagada',
    'Ikotun', 'Ifako-Gbagada', 'Isolo', 'Alapere-Ketu', 'Ojo', 'Agege', 'Fola-Agoro-Yaba', 'Oregun-Ikeja',
    'Ikota-Lekki', 'Ikate-Lekki', 'Ilupeju', 'Phase2-Lekki', 'Oko-Oba-Agege', 'Maryland', 'Mende-Maryland',
    'Ikosi-Ketu', 'Abule-Egba', 'Ibeju-Lekki', 'GRA-Ikeja', 'Lagos-Island', 'Akowonjo', 'Apapa', 'Idimu',
    'Opebi-Ikeja', 'Igando', 'Oworonshoki-Gbagada', 'Akoka-Yaba', 'Alimosho', 'Sabo-Yaba', 'Agungi-Lekki',
    'Oshodi', 'Ayobo', 'Iju-Ishaga-Agege', 'Allen-Avenue-Ikeja', 'Okota', 'Akesan', 'Cement-Agege',
    'Phase1-Gbagada', 'Anthony-Village-Maryland', 'Fagba-Agege', 'Shangisha-Ketu', 'Mile12-Ketu', 'Orile-Agege',
    'Jibowu-Yaba', 'Ejigbo', 'Adekunle-Yaba', 'Bariga', 'Mushin', 'Epe', 'Abule-Ijesha-Yaba', 'Badagry',
    'VGC-Lekki', 'Abule-Oja-Yaba', 'Ebute-Metta-Yaba', 'Iwaya-Yaba', 'Ifako-Agege', 'Ilasamaja-Mushin',
    'Marina-Lagos-Island', 'Iganmu-Orile', 'Obalende-Lagos-Island'
])

# House features input
bedroom = st.sidebar.slider("Number of Bedrooms", 1, 10, 3)
bathroom = st.sidebar.slider("Number of Bathrooms", 1, 10, 2)
toilet = st.sidebar.slider("Number of Toilets", 1, 10, 2)
House_type = st.sidebar.selectbox("House Type", ['Flat', 'House', 'Duplex', 'Bungalow'])
Island = st.sidebar.selectbox('Island:', ['Yes', 'No'])

# Predict price on button click
predict_button = st.sidebar.button('Predict Price')

# Create a spinner in the main st context
spinner = st.spinner('Predicting...')

# Display the rolling predicting sign under the "Predict Price" button
if predict_button:
    with spinner:
        time.sleep(2)  # Simulating prediction time

        # Prediction function
        def predict(bedroom, bathroom, toilet, Location, House_type, Island):
            new_location = {'Lekki': 0, 'Ikeja': 1, 'Ipaja': 2, 'Ojodu': 3, 'Surulere': 4, 'Onike-Yaba': 5,
                            'Ketu': 6, 'Phase1-Lekki': 7, 'Shomolu': 8, 'Amuwo-Odofin': 9, 'Ikotun-Igando': 10,
                            'Ajah': 11, 'Sangotedo': 12, 'Ojota': 13, 'Ogba': 14, 'Ogudu': 15, 'Phase1-Magodo': 16,
                            'Ikorodu': 17, 'Alagomeji-Yaba': 18, 'Victoria-Island': 19, 'Osapa-London-Lekki': 20,
                            'Ago-Palace-Okota': 21, 'Medina-Gbagada': 22, 'Yaba': 23, 'Ikoyi': 24, 'Gbagada': 25,
                            'Egbeda': 26, 'Soluyi-Gbagada': 27, 'Chevron-Lekki': 28, 'Phase2-Magodo': 29,
                            'Phase2-Gbagada': 30, 'Ikotun': 31, 'Ifako-Gbagada': 32, 'Isolo': 33, 'Alapere-Ketu': 34,
                            'Ojo': 35, 'Agege': 36, 'Fola-Agoro-Yaba': 37, 'Oregun-Ikeja': 38, 'Ikota-Lekki': 39,
                            'Ikate-Lekki': 40, 'Ilupeju': 41, 'Phase2-Lekki': 42, 'Oko-Oba-Agege': 43, 'Maryland': 44,
                            'Mende-Maryland': 45, 'Ikosi-Ketu': 46, 'Abule-Egba': 47, 'Ibeju-Lekki': 48,
                            'GRA-Ikeja': 49, 'Lagos-Island': 50, 'Akowonjo': 51, 'Apapa': 52, 'Idimu': 53,
                            'Opebi-Ikeja': 54, 'Igando': 55, 'Oworonshoki-Gbagada': 56, 'Akoka-Yaba': 57, 'Alimosho': 58,
                            'Sabo-Yaba': 59, 'Agungi-Lekki': 60, 'Oshodi': 61, 'Ayobo': 62, 'Iju-Ishaga-Agege': 63,
                            'Allen-Avenue-Ikeja': 64, 'Okota': 65, 'Akesan': 66, 'Cement-Agege': 67, 'Phase1-Gbagada': 68,
                            'Anthony-Village-Maryland': 69, 'Fagba-Agege': 70, 'Shangisha-Ketu': 71, 'Mile12-Ketu': 72,
                            'Orile-Agege': 73, 'Jibowu-Yaba': 74, 'Ejigbo': 75, 'Adekunle-Yaba': 76, 'Bariga': 77,
                            'Mushin': 78, 'Epe': 79, 'Abule-Ijesha-Yaba': 80, 'Badagry': 81, 'VGC-Lekki': 82,
                            'Abule-Oja-Yaba': 83, 'Ebute-Metta-Yaba': 84, 'Iwaya-Yaba': 85, 'Ifako-Agege': 86,
                            'Ilasamaja-Mushin': 87, 'Marina-Lagos-Island': 88, 'Iganmu-Orile': 89, 'Obalende-Lagos-Island': 90}

            for i in new_location:
                if Location == i:
                    Location = new_location[i]

            if House_type == 'Flat':
                House_type = 1
            elif House_type == 'House':
                House_type = 2
            elif House_type == 'Duplex':
                House_type = 3
            elif House_type == 'Bungalow':
                House_type = 4

            # Convert input features to the correct data types
            bedroom = int(bedroom)
            bathroom = int(bathroom)
            toilet = int(toilet)
            Location = int(Location)  # Assuming the location is numerical
            House_type = int(House_type)  # Assuming the house type is numerical
            Island = 1 if Island == 'Yes' else 0  # Convert 'Yes' to 1, 'No' to 0

            # Ensure the order of features matches the order during training
            prediction = xgb_model.predict(
                pd.DataFrame([[Location, bedroom, bathroom, toilet, House_type, Island]],
                             columns=['LOCATION', 'BEDROOMS', 'BATHROOMS', 'TOILETS', 'HOUSE_TYPE', 'ISLAND'])
            )
            return prediction

        # Get prediction
        price = predict(bedroom, bathroom, toilet, Location, House_type, Island)

        # Display the result
        st.sidebar.write('The Predicted Rent of the House is: ₦{:,.2f}'.format(price[0]))

        