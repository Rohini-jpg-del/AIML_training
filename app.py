# import streamlit as st
# st.title("Wine Quality Prediction")
# age=st.number_input("Enter the age of wine (in years):",min_value=1)
# weight=st.number_input("Enter the weight of the wine (in grams):",min_value=)
# if st.button("Predict Quality"):
#     import pandas as pd 
#     prediction=model.predict([[age,weight]])
#     st.write("The predicted quality of the wine is:",prediction[0])

# import streamlit as st
# import pandas as pd
# import joblib

# # Load pipeline components
# hpp = joblib.load("house_rent_model.pkl")

# encoder = hpp["encoder"]
# scaler = hpp["scaler"]
# model = hpp["model"]

# st.set_page_config(page_title="House Rent Prediction", page_icon="🏠")

# st.title("🏠 House Rent Prediction")
# st.write("Enter house details to predict monthly rent.")

# # User Inputs
# bhk = st.number_input("BHK", min_value=1, max_value=20, value=2)

# size = st.number_input(
#     "Size (sq.ft)",
#     min_value=100,
#     max_value=10000,
#     value=1200
# )

# area_type = st.selectbox(
#     "Area Type",
#     ["Super Area", "Carpet Area", "Built Area"]
# )

# city = st.text_input("City", "Delhi")

# furnishing = st.selectbox(
#     "Furnishing Status",
#     ["Furnished", "Semi-Furnished", "Unfurnished"]
# )

# tenant = st.selectbox(
#     "Tenant Preferred",
#     ["Bachelors", "Family", "Bachelors/Family"]
# )

# bathroom = st.number_input(
#     "Bathrooms",
#     min_value=1,
#     max_value=20,
#     value=2
# )

# contact = st.selectbox(
#     "Point of Contact",
#     ["Contact Owner", "Contact Agent", "Contact Builder"]
# )

# # Prediction
# if st.button("Predict Rent"):

#     new_house = pd.DataFrame({
#         "BHK": [bhk],
#         "Size": [size],
#         "Area Type": [area_type],
#         "City": [city],
#         "Furnishing Status": [furnishing],
#         "Tenant Preferred": [tenant],
#         "Bathroom": [bathroom],
#         "Point of Contact": [contact]
#     })

#     try:
#         encoded = encoder.transform(new_house)
#         scaled = scaler.transform(encoded)

#         prediction = model.predict(scaled)[0]

#         st.success(
#             f"Predicted Monthly Rent: ₹ {prediction:,.0f}"
#         )

#     except Exception as e:
#         st.error(f"Prediction Error: {e}")
import streamlit as st
import category_encoders as ce
from sklearn.preprocessing import MinMaxScaler
st.title("House Rent Prediction")
import joblib
import pandas as pd
model = joblib.load("house_rent_model.pkl")

BHK = st.number_input("Enter the number of BHK:", min_value=1, max_value=10, step=1)
Size = st.number_input("Enter the size of the house in square feet:", min_value=100, max_value=10000, step=10)
Area_Type = st.selectbox("Select the area type:", ["Super Area", "Carpet Area", "Built Area"])
City = st.selectbox("Select the city:", ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata"])
Furnishing_Status = st.selectbox("Select the furnishing status:", ["Furnished", "Semi-Furnished", "Unfurnished"])
Tenant_Preferred = st.selectbox("Select the tenant preferred:", ["Family", "Bachelors", "Any"])
Bathroom = st.number_input("Enter the number of bathrooms:", min_value=1, max_value=10, step=1)
Point_of_Contact = st.selectbox("Select the point of contact:", ["Contact Owner", "Contact Agent", "Contact Builder"])
input=pd.DataFrame({
    'BHK':[BHK],
     'Size': [Size], 
     'Area Type': [Area_Type], 
     'City': [City], 
     'Furnishing Status': [Furnishing_Status],
     'Tenant Preferred': [Tenant_Preferred],
        'Bathroom': [Bathroom],
        'Point of Contact': [Point_of_Contact]
})
e=model["encoder"]
s=model["scaler"]
m=model["model"]
new_encoded = e.transform(input)
new_scaled = s.transform(new_encoded)

if st.button("Predict Rent"):
    prediction = m.predict(new_scaled)
    st.write(f"The predicted rent is: {prediction[0]}")