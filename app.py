#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Property Sale Price Predictor",
    page_icon="🏠",
    layout="centered"
)


# ---------------------------------------------------------
# Loading trained model
# ---------------------------------------------------------

model = joblib.load("Model\\property_price_model.pkl")


# ---------------------------------------------------------
# Property description feature extraction
# ---------------------------------------------------------

description_patterns = {
    "Has_pool": r"\bpool\b|swimming pool",
    "Has_garage": r"\bgarage\b|lock[- ]?up garage",
    "Has_balcony": r"\bbalcony\b",
    "Has_garden": r"\bgarden\b",
    "Has_views": r"\bviews?\b|water view|city view",
    "Has_renovated": r"\brenovat(?:ed|ion|ing)\b|refurbished|updated",
    "Has_aircon": r"air[- ]?conditioning|air[- ]?con",
    "Has_ensuite": r"\bensuite\b|en[- ]?suite",
    "Has_builtins": r"built[- ]?in|built in wardrobes?",
    "Has_courtyard": r"\bcourtyard\b"
}


def extract_description_features(description):

    description = str(description).lower()

    features = {}

    for feature, pattern in description_patterns.items():

        features[feature] = int(
            pd.Series([description])
            .str.contains(
                pattern,
                regex=True,
                na=False
            )
            .iloc[0]
        )

    return features


# ---------------------------------------------------------
# Application title
# ---------------------------------------------------------

st.title("🏠 Property Sale Price Predictor")

st.write(
    """
    Enter the characteristics of a residential property to
    obtain an estimated sale price from the trained machine
    learning model.
    """
)


# ---------------------------------------------------------
# User inputs
# ---------------------------------------------------------

st.header("Property Information")

locality = st.selectbox(
    "Locality",
    [
        "Mosman",
        "Parramatta",
        "Blacktown"
    ]
)

property_type = st.selectbox(
    "Property Type",
    [
        "Apartment / Unit / Flat",
        "House",
        "Townhouse"
    ]
)

bed = st.number_input(
    "Bedrooms",
    min_value=0,
    max_value=10,
    value=3,
    step=1
)

bath = st.number_input(
    "Bathrooms",
    min_value=0,
    max_value=10,
    value=2,
    step=1
)

parking = st.number_input(
    "Parking Spaces",
    min_value=0,
    max_value=10,
    value=1,
    step=1
)

distance_from_school = st.number_input(
    "Distance from School (km)",
    min_value=0.0,
    max_value=20.0,
    value=1.0,
    step=0.1
)

sale_year = st.number_input(
    "Sale Year",
    min_value=2000,
    max_value=2030,
    value=2026,
    step=1
)

sale_month = st.number_input(
    "Sale Month",
    min_value=1,
    max_value=12,
    value=1,
    step=1
)

sale_quarter = ((sale_month - 1) // 3) + 1

property_description = st.text_area(
    "Property Description",
    placeholder=(
        "Example: Renovated family home with garden, "
        "balcony and garage..."
    )
)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

if st.button("Predict Sale Price"):

    # Extract description features
    description_features = extract_description_features(
        property_description
    )

    # Engineered features
    if bed != 0:
        bath_per_bed = bath / bed
    else:
        bath_per_bed = np.nan

    bed_bath_total = bed + bath

    # Construct input dataframe
    input_data = {
        "Bed": bed,
        "Bath": bath,
        "Parking": parking,
        "Distance_from_school_km": distance_from_school,
        "Sale_year": sale_year,
        "Sale_month_num": sale_month,
        "Sale_quarter": sale_quarter,
        "Bath_per_Bed": bath_per_bed,
        "Bed_Bath_Total": bed_bath_total,
        "Locality": locality,
        "Property_type": property_type
    }

    input_data.update(description_features)

    input_df = pd.DataFrame([input_data])

    # Generate prediction
    prediction = model.predict(input_df)[0]

    # Display result
    st.success(
        f"Estimated Sale Price: ${prediction:,.0f}"
    )

    st.info(
        """
        This prediction is an estimate based on the characteristics
        provided and should not be interpreted as a professional
        property valuation.
        """
    )

