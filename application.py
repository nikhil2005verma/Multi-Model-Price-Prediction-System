import numpy as np
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.sidebar.title('select an Option')
user_menu=st.sidebar.radio(
    " ",
    ('Car Price Prediction','Bangalore House price Prediction','Laptop price Prediction')
)

if user_menu=='Car Price Prediction':
    st.sidebar.markdown(
    "<hr style='border:2px solid red'>",
    unsafe_allow_html=True
    )
    st.sidebar.title("Car Price Prediction")
    df=pd.read_csv('cleaned_car.csv')

    # Load trained model
    model = pickle.load(open("carLinearRegressionModel.pkl", "rb"))

    st.title("🚗 welcome to Car Price Predicto")
    st.divider()

    st.title("Enter car details to predict price")

    # Inputs
    car=df['name'].unique()
    name = st.selectbox("Car Name:",car)

    Year=[1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989,
 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
 2020, 2021, 2022, 2023, 2024, 2025, 2026]

    year = st.selectbox("Select Year of Purchase:",Year)

    company=df['company'].unique()
    company.sort()

    company = st.selectbox(" Select the Company:",company)

    km=np.arange(10000,200000,1000)

    kms_driven = st.selectbox("Select the Number of km that the car has travelled:",km)

    fuel=df['fuel_type'].unique()
    fuel.sort()

    fuel_type = st.selectbox("Select Fuel Type:",fuel)

    # Predict button
    if st.button("Predict Price"):

        # Create DataFrame
        input_data = pd.DataFrame({
            "name": [name],
            "year": [year],
            "company": [company],
            "kms_driven": [kms_driven],
            "fuel_type": [fuel_type]
        })

        # Prediction
        try:
            prediction = model.predict(input_data)

            st.success(f"Estimated Price: ₹ {int(prediction[0]):,}")


            def assign_owner(km):
                if kms_driven < 30000:
                    return str(1)+'st'
                elif kms_driven < 80000:
                    return str(2)+'nd'
                else:
                    return str(3)+'rd'


            owner = assign_owner(kms_driven)
            st.title(owner + ' owner of this car')

        except Exception as e:
            st.error("Model error: Check input format or encoding")
            st.write(e)




if user_menu =='Bangalore House price Prediction':
    st.sidebar.markdown(
        "<hr style='border:2px solid red'>",
        unsafe_allow_html=True
    )
    st.sidebar.title("Bangalore House Price Prediction")
    st.title("🏠 welcome to Bangalore House Price Predictor")
    st.divider()


    # Load model
    model = pickle.load(open("houseLinearregrression.pkl", "rb"))

    df1=pd.read_csv('cleaned_house.csv')
    location=df1['location'].unique()
    location.sort()
    location = st.selectbox("Select the Location:",location)

    sqft=df1['total_sqft'].unique()
    sqft.sort()
    total_sqft = st.selectbox("Select square feet:",sqft)

    bath=df1['bath'].unique()
    bath.sort()
    bath=bath.astype(int)
    bath = st.selectbox("Select number of Bathrooms:",bath)

    bhk=df1['bhk'].unique()
    bhk.sort()
    bhk = st.selectbox("Select Bhk:",bhk)

    # Predict
    if st.button("Predict Price"):

        try:
            # Create dataframe
            input_df = pd.DataFrame({
                "location": [location],
                "total_sqft": [total_sqft],
                "bath": [bath],
                "bhk": [bhk]
            })

            # Prediction
            prediction = model.predict(input_df)

            st.success(f"💰 Estimated Price: ₹ {round(prediction[0], 2)} Lakhs")

        except Exception as e:
            st.error("Error in prediction. Check model or preprocessing.")
            st.write(e)






if user_menu =='Laptop price Prediction':
    st.sidebar.markdown(
        "<hr style='border:2px solid red'>",
        unsafe_allow_html=True
    )
    st.sidebar.title("Laptop Price Prediction")
    st.title("💻 welcome to Laptop Price Predictor")
    st.divider()

    # import the model and data
    pipe = pickle.load(open('laptop_pipe.pkl', 'rb'))
    df = pickle.load(open('laptop_df.pkl', 'rb'))

    company = st.selectbox('Brand', df['Company'].unique())

    # type of laptop
    type = st.selectbox('Type', df['TypeName'].unique())

    # Ram
    ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

    # weight
    weight = st.number_input('Weight of the Laptop')

    # Touchscreen
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

    # IPS
    ips = st.selectbox('IPS', ['No', 'Yes'])

    # screen size
    screen_size = st.slider('Scrensize in inches', 10.0, 18.0, 13.0)

    # resolution
    resolution = st.selectbox('Screen Resolution',
                              ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600',
                               '2560x1440', '2304x1440'])

    # cpu
    cpu = st.selectbox('CPU', df['Cpu brand'].unique())

    hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

    ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

    gpu = st.selectbox('GPU', df['Gpu brand'].unique())

    os = st.selectbox('OS', df['os'].unique())

    if st.button('Predict Price'):
        # query
        ppi = None
        if touchscreen == 'Yes':
            touchscreen = 1
        else:
            touchscreen = 0

        if ips == 'Yes':
            ips = 1
        else:
            ips = 0

        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
        query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

        query = query.reshape(1, 12)
        st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))

