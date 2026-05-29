import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_icon='🏠', page_title='Bengaluru House Price Predictor', layout='centered')

st.title('🏠 Bengaluru House Price Prediction')
st.markdown('Predict the price of a house in Bengaluru based on its features.')

@st.cache_resource
def load_model():
    return joblib.load('RF_model.joblib')

@st.cache_data
def load_locations():
    df = pd.read_csv('With_location.csv')
    locs = df[['location', 'encoded_loc']].drop_duplicates().sort_values('location')
    return locs

model = load_model()
locations_df = load_locations()

location_names = locations_df['location'].tolist()
location_encoding = dict(zip(locations_df['location'], locations_df['encoded_loc']))

st.subheader('Enter the details:')

col1, col2 = st.columns(2)

with col1:
    size = st.number_input('BHK (Size)', min_value=1, max_value=6, value=2, step=1)
    total_sqft = st.number_input('Total Square Feet', min_value=276.0, max_value=100000.0, value=1200.0, step=50.0)
    bath = st.number_input('Number of Bathrooms', min_value=1, max_value=7, value=2, step=1)

with col2:
    balcony = st.number_input('Number of Balconies', min_value=0, max_value=3, value=1, step=1)
    location = st.selectbox('Location', options=location_names)

if st.button('💰 Predict Price', type='primary', use_container_width=True):
    encoded_loc = location_encoding[location]
    input_df = pd.DataFrame([{
        'size': size,
        'total_sqft': total_sqft,
        'bath': bath,
        'balcony': balcony,
        'encoded_loc': encoded_loc
    }])

    estimator = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
    prediction = estimator.predict(input_df)[0]

    st.success(f'## Predicted Price: ₹ {prediction:.2f} Lakhs')
    st.info(f'That is approximately ₹ {prediction * 100000:,.0f}')
