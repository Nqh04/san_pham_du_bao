import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model and encoders
# In a real app, handle potential errors during loading
try:
    model = joblib.load('random_forest_model.joblib')
    le_product_detail = joblib.load('le_product_detail.joblib')
    # Assuming you also saved feature encoders in feature_encoders_dict.joblib
    # feature_encoders_dict = joblib.load('feature_encoders_dict.joblib')
    st.success("Model and encoders loaded successfully!")
except Exception as e:
    st.error(f"Error loading model or encoders: {e}")
    st.info("Please ensure 'random_forest_model.joblib' and 'le_product_detail.joblib' (and potentially feature_encoders_dict.joblib) are in the same directory.")
    # Exit or handle the error appropriately if loading fails
    st.stop() # Stop the app if essential files are missing


# Define the prediction function (copied from previous steps and adapted)
# Ensure this function matches the preprocessing and feature engineering done during training
def predict_product_detail_revised(model, target_encoder, input_features):
    """
    Predicts product_detail based on input features using a trained model.

    Args:
        model: The trained scikit-learn model.
        target_encoder: The LabelEncoder object for the target variable 'product_detail'.
        input_features: A dictionary containing input feature values.
                        Keys should match the original column names used for feature engineering.


    Returns:
        The predicted product_detail as a string, or None if prediction fails.
    """
    processed_features = {}

    # --- Feature Engineering and Preprocessing matching training ---
    # This part needs to replicate the preprocessing steps from the training notebook

    # Handle categorical features (need to use the same encoders as training)
    # If you saved feature_encoders_dict, load and use it here.
    # For now, assuming we need to re-encode based on input values if encoders weren't saved for features.
    # A robust app would save and load all necessary encoders.

    # Example of re-encoding (less robust than loading saved encoders)
    # This assumes the input categories will match the training categories exactly.
    # A better approach is to save and load feature_encoders_dict
    # For demonstration, let's manually handle the encoding based on the features used in the final training step
    # Referencing the feature_columns list from the last successful training run:
    # feature_columns = ['Age', 'Income', 'Total_Bill', 'Gender_encoded', 'Occupation_encoded',
    #                    'Season_encoded', 'product_category_encoded',
    #                    'product_type_encoded', 'Size_encoded', 'unit_price', 'store_location_encoded',
    #                    'year', 'month', 'day', 'day_of_week', 'hour']

    # To correctly implement this, you would need to save ALL LabelEncoders used for features
    # and load them into feature_encoders_dict here.
    # For now, let's create a placeholder and note this is crucial.
    # feature_encoders_dict = {} # Load saved feature encoders here!

    # Replicate date/time feature extraction (assuming original date/time strings are in input_features)
    try:
        date_obj = pd.to_datetime(input_features.get('transaction_date'))
        input_features['year'] = date_obj.year
        input_features['month'] = date_obj.month
        input_features['day'] = date_obj.day
        input_features['day_of_week'] = date_obj.dayofweek # Monday=0, Sunday=6
    except Exception as e:
        st.warning(f"Could not parse transaction_date: {e}. Date-derived features may be missing.")
        # Decide how to handle missing derived features - maybe return None or use default values
        return None # Or handle missing features


    try:
         time_parts = str(input_features.get('transaction_time')).split(':')
         if len(time_parts) >= 1:
             input_features['hour'] = int(time_parts[0])
         else:
             st.warning(f"Could not parse transaction_time. Hour feature will be missing.")
             return None # Or handle missing features
    except Exception as e:
         st.warning(f"Could not parse transaction_time: {e}. Hour feature will be missing.")
         return None # Or handle missing features


    # Manual encoding for demonstration - replace with loading saved encoders
    # This is error-prone if input categories are not seen during training
    # You need to save and load 'feature_encoders_dict' from your training notebook
    # For now, let's assume the encoders are available globally if run in the same environment
    # In a standalone app, this is NOT the way to do it.
    try:
        input_features['Gender_encoded'] = df['Gender'].unique().tolist().index(input_features['Gender']) # Example - replace with proper encoder
        input_features['Occupation_encoded'] = df['Occupation'].unique().tolist().index(input_features['Occupation']) # Example
        input_features['Season_encoded'] = df['Season'].unique().tolist().index(input_features['Season']) # Example
        input_features['product_category_encoded'] = df['product_category'].unique().tolist().index(input_features['product_category']) # Example
        input_features['product_type_encoded'] = df['product_type'].unique().tolist().index(input_features['product_type']) # Example
        input_features['Size_encoded'] = df['Size'].unique().tolist().index(input_features['Size']) # Example
        input_features['store_location_encoded'] = df['store_location'].unique().tolist().index(input_features['store_location']) # Example
        # Also need to handle store_id if it was used as a feature and is numerical
        # input_features['store_id'] = int(input_features['store_id']) # Ensure numerical
        input_features['unit_price'] = float(input_features['unit_price'])
        input_features['Total_Bill'] = float(input_features['Total_Bill'])
        input_features['Age'] = int(input_features['Age'])
        input_features['Income'] = float(input_features['Income'])


    except Exception as e:
         st.error(f"Error during manual feature encoding: {e}. Please check input values.")
         st.info("Note: For a robust app, save and load feature encoders from training.")
         return None


    # Create a dictionary with all model feature names, mapping to values from input_features
    # This list of feature names MUST match the exact list and order used during training
    # Get the feature names from the trained model if possible, or save them during training
    # Assuming the feature names from the last training run are available (replace with loading saved list)
    # feature_columns = ['Age', 'Income', 'Total_Bill', 'Gender_encoded', 'Occupation_encoded',
    #                    'Season_encoded', 'product_category_encoded',
    #                    'product_type_encoded', 'Size_encoded', 'unit_price', 'store_location_encoded',
    #                    'year', 'month', 'day', 'day_of_week', 'hour']
    # For a robust app, save and load the exact list of model feature names!
    # For now, let's assume the global 'X.columns' from the last successful run is available
    # In a standalone app, save X.columns during training and load it here.
    model_feature_names = X.columns.tolist() # Replace with loading saved list

    input_df_data = {}
    for feature_name in model_feature_names:
         if feature_name in input_features:
              input_df_data[feature_name] = input_features[feature_name]
         else:
              st.error(f"Error: Feature '{feature_name}' expected by model is missing after processing input.")
              return None # Indicate failure


    # Convert the dictionary to a DataFrame, ensuring correct column order
    # The order of columns in the DataFrame must match the order used during training
    ordered_values = [input_df_data.get(col) for col in model_feature_names]
    input_df = pd.DataFrame([ordered_values], columns=model_feature_names)

    # Ensure data types are correct (matching training data types is ideal)
    # Attempt to convert numerical columns to appropriate types
    # This list of numerical/derived features needs to match the training data
    numerical_derived_cols = ['Age', 'Income', 'Total_Bill', 'unit_price',
                              'year', 'month', 'day', 'day_of_week', 'hour'] # Add other numerical/derived features used
    # Add encoded categorical features to the list to ensure they are treated numerically by the model
    encoded_cols = [col for col in model_feature_names if col.endswith('_encoded')]
    numerical_derived_cols.extend(encoded_cols)

    for col in input_df.columns:
        if col in numerical_derived_cols:
            try:
                input_df[col] = pd.to_numeric(input_df[col])
            except ValueError as e:
                st.warning(f"Could not convert column '{col}' to numeric: {e}. Data type mismatch might occur.")
        # Handle 'store_id' if used - ensure it's numeric
        if col == 'store_id':
             try:
                 input_df[col] = pd.to_numeric(input_df[col])
             except ValueError as e:
                  st.warning(f"Could not convert column '{col}' to numeric: {e}. Data type mismatch might occur.")


    # Make prediction
    if input_df.isnull().any().any():
         st.error("Error: Input DataFrame contains missing values after processing. Cannot make prediction.")
         st.dataframe(input_df) # Show the problematic input df
         return None

    try:
        prediction_encoded = model.predict(input_df)[0]
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        st.dataframe(input_df) # Show the input df that caused prediction error
        return None

    # Inverse transform the prediction
    try:
        predicted_product_detail = target_encoder.inverse_transform([prediction_encoded])[0]
        return predicted_product_detail
    except Exception as e:
        st.error(f"Error during inverse transformation: {e}")
        return None


# --- Streamlit App UI ---

st.title('Product Detail Prediction App')
st.write('Enter the transaction details to predict the product detail.')

# Collect user input for each feature used by the model
# Refer to the list of features used in the final training run (X.columns)
# You need input fields for ALL features in X.columns except the derived ones (year, month, day, etc.)
# For derived features, you need input for the original columns they were derived from (transaction_date, transaction_time)

st.sidebar.header('Input Features')

# List of original columns from which features were derived or used directly
# This list should match the columns used as input in the training notebook
# Based on the last training run using all features:
# Original columns needed for input:
# 'Age', 'Income', 'Total_Bill', 'Gender', 'Occupation', 'Season',
# 'product_category', 'product_type', 'Size', 'unit_price', 'store_location',
# 'transaction_date', 'transaction_time' (for deriving year, month, day, hour, day_of_week)
# 'store_id' (if used as a direct feature)

input_features = {}

# Numerical Features (original)
numerical_cols_input = ['Age', 'Income', 'Total_Bill', 'unit_price']
for col in numerical_cols_input:
     # Add input widgets for numerical features
     # You might want to set min/max values based on your training data
     input_features[col] = st.sidebar.number_input(f'{col}', value=0.0) # Adjust default value and step as needed

# Categorical Features (original strings)
categorical_cols_input = ['Gender', 'Occupation', 'Season', 'product_category', 'product_type', 'Size', 'store_location']
# For categorical features, it's best to provide a selectbox with the categories seen during training
# If you saved feature_encoders_dict, you can get the classes from there.
# For now, let's use text input as a placeholder, but selectbox is better.
# In a real app, load the categories from saved encoders or a list saved during training.
# For demonstration, let's use dummy options. Replace with actual categories.
dummy_options = ['Category1', 'Category2', 'Category3'] # Replace with actual categories loaded from training

for col in categorical_cols_input:
     # In a real app, load actual options from saved encoders or a list
     # options = feature_encoders_dict[col].classes_.tolist() if col in feature_encoders_dict else dummy_options
     input_features[col] = st.sidebar.text_input(f'{col}', 'Enter category') # Replace with st.selectbox using actual options

# Date and Time Inputs (original strings for derivation)
input_features['transaction_date'] = st.sidebar.text_input('Transaction Date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)', '2023-01-01')
input_features['transaction_time'] = st.sidebar.text_input('Transaction Time (HH:MM:SS or HH:MM)', '12:00:00')

# Handle 'store_id' if it was used as a direct feature (numerical)
# If 'store_id' was in the final X.columns and was numerical, add an input for it.
# Check if 'store_id' is in the list of features used by the model (model_feature_names)
# model_feature_names = X.columns.tolist() # Load this list from training!
# if 'store_id' in model_feature_names:
#     input_features['store_id'] = st.sidebar.number_input('store_id', min_value=0, step=1, value=1) # Adjust min/default/max


# Prediction button
if st.button('Predict Product Detail'):
    # Ensure model and le_product_detail were loaded successfully
    if 'model' in globals() and 'le_product_detail' in globals():
        # Gather inputs into the input_features dictionary (already done above)
        # Call the prediction function
        predicted_detail = predict_product_detail_revised(model, le_product_detail, input_features)

        if predicted_detail:
            st.subheader('Prediction')
            st.success(f'Predicted Product Detail: {predicted_detail}')
        else:
            st.error('Prediction failed. Please check the input values and console for errors.')
    else:
        st.error("Model or encoders not loaded. Please check the file paths.")
