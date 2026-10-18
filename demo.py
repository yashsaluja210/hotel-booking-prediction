import joblib
import streamlit as st
import pandas as pd

# Load the saved model, scaler, and columns for Booking
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("columns.pkl")

# App title and Description
st.title("🏨 Hotel Reservation Status Predictor")
st.write("Enter the booking details below to predict if the booking is likely to be Canceled or Not Canceled.")

# Collect User Inputs using a clean form layout
with st.form("booking_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        number_of_adults = st.number_input("Number of Adults", min_value=0, max_value=10, value=2)
        number_of_children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        number_of_weekend_nights = st.number_input("Number of Weekend Nights", min_value=0, max_value=10, value=1)
        number_of_week_nights = st.number_input("Number of Week Nights", min_value=0, max_value=30, value=2)
        lead_time = st.number_input("Lead Time (Days)", min_value=0, max_value=500, value=50)
        average_price = st.number_input("Average Room Price ($)", min_value=0.0, max_value=1000.0, value=100.0)
        special_requests = st.number_input("Number of Special Requests", min_value=0, max_value=10, value=0)
        car_parking_space = st.selectbox("Car Parking Space Required", [0, 1])

    with col2:
        repeated = st.selectbox("Is Repeated Guest?", [0, 1])
        p_c = st.number_input("Previous Cancellations (P-C)", min_value=0, max_value=50, value=0)
        p_not_c = st.number_input("Previous Bookings Not Canceled (P-not-C)", min_value=0, max_value=50, value=0)
        reservation_year = st.number_input("Reservation Year", min_value=2015, max_value=2030, value=2018)
        reservation_month = st.number_input("Reservation Month", min_value=1, max_value=12, value=5)
        reservation_week = st.number_input("Reservation Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=2)
        
        type_of_meal = st.selectbox("Type of Meal Plan", ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"])
        room_type = st.selectbox("Room Type", ["Room_Type 1", "Room_Type 2", "Room_Type 3", "Room_Type 4", "Room_Type 5", "Room_Type 6", "Room_Type 7"])
        market_segment_type = st.selectbox("Market Segment Type", ["Online", "Offline", "Corporate", "Complementary", "Aviation"])

    submit_button = st.form_submit_button("Predict Booking Status")

# Prediction logic when button is clicked
if submit_button:
    # Feature engineering matching notebook steps
    total_nights = number_of_weekend_nights + number_of_week_nights
    total_family_member = number_of_adults + number_of_children

    # Build input dictionary
    input_data = {
        "number of adults": number_of_adults,
        "number of children": number_of_children,
        "number of weekend nights": number_of_weekend_nights,
        "number of week nights": number_of_week_nights,
        "car parking space": car_parking_space,
        "lead time": lead_time,
        "repeated": repeated,
        "P-C": p_c,
        "P-not-C": p_not_c,
        "average price": average_price,
        "special requests": special_requests,
        "reservation_year": reservation_year,
        "reservation_month": reservation_month,
        "reservation_week": reservation_week,
        "total_nights": total_nights,
        "total_family_member": total_family_member,
        "type of meal": type_of_meal,
        "room type": room_type,
        "market segment type": market_segment_type
    }

    # Convert to DataFrame
    df_input = pd.DataFrame([input_data])

    # Apply get_dummies matching training pipeline with drop_first=True
    df_input = pd.get_dummies(
        df_input,
        columns=["type of meal", "room type", "market segment type"],
        drop_first=True,
        dtype=int
    )

    # Reindex to ensure exact column match with training columns
    df_input = df_input.reindex(columns=model_columns, fill_value=0)

    
    prediction = model.predict(df_input)[0]

    # Display Result (0 = Not Canceled, 1 = Canceled)
    if prediction == 1:
        st.error("⚠️ Prediction: This booking is likely to be **Canceled**.")
    else:
        st.success("✅ Prediction: This booking is **Not Canceled**.")