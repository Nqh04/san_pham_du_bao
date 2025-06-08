import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained Logistic Regression model (full features) and encoder
model = joblib.load('logistic_regression_model_full.joblib')
encoder = joblib.load('onehot_encoder_full.joblib')

# Define the selected features that the app will take input for (based on original columns)
input_features_original = ['Total_Bill', 'Age', 'Income', 'Season', 'store_location', 'Gender', 'Occupation', 'transaction_qty', 'unit_price', 'product_type', 'product_detail', 'Size']


# Get the original categories from the encoder to create input widgets
original_categories = encoder.categories_
store_locations = original_categories[0] # Assuming store_location was the first column encoded
product_categories_original = original_categories[1] # Original product categories
product_types = original_categories[2] # Assuming product_type was the third column encoded
product_details = original_categories[3] # Assuming product_detail was the fourth column encoded
sizes = original_categories[4] # Assuming Size was the fifth column encoded
genders = original_categories[5] # Assuming Gender was the sixth column encoded
occupations = original_categories[6] # Assuming Occupation was the seventh column encoded
seasons = original_categories[7] # Assuming Season was the eighth column encoded

# Streamlit app title
st.title("Dự báo Product Category")

# User input for features
st.sidebar.header("Nhập thông tin khách hàng")

# Input fields based on the original input features
input_data = {}

# Numerical inputs
input_data['transaction_qty'] = st.sidebar.number_input("Số lượng giao dịch", min_value=1, value=1)
input_data['unit_price'] = st.sidebar.number_input("Đơn giá", min_value=0, value=30000)
input_data['Total_Bill'] = st.sidebar.number_input("Tổng hóa đơn", min_value=0, value=100000)
input_data['Age'] = st.sidebar.number_input("Tuổi", min_value=0, max_value=120, value=30)
input_data['Income'] = st.sidebar.number_input("Thu nhập", min_value=0, value=5000000)


# Categorical inputs (using original categories)
selected_season = st.sidebar.selectbox("Mùa", seasons)
selected_store_location = st.sidebar.selectbox("Địa điểm cửa hàng", store_locations)
selected_gender = st.sidebar.selectbox("Giới tính", genders)
selected_occupation = st.sidebar.selectbox("Nghề nghiệp", occupations)
selected_product_type = st.sidebar.selectbox("Loại sản phẩm", product_types)
selected_product_detail = st.sidebar.selectbox("Chi tiết sản phẩm", product_details)
selected_size = st.sidebar.selectbox("Kích cỡ", sizes)


# Button to trigger prediction
if st.sidebar.button("Dự báo"):
    # Create a DataFrame from the numerical input data
    input_df_numerical = pd.DataFrame([{
        'transaction_qty': input_data['transaction_qty'],
        'unit_price': input_data['unit_price'],
        'Total_Bill': input_data['Total_Bill'],
        'Age': input_data['Age'],
        'Income': input_data['Income']
    }])

    # Create a DataFrame from the categorical input data based on original column names
    input_df_categorical_original = pd.DataFrame([{
        'Season': selected_season,
        'store_location': selected_store_location,
        'Gender': selected_gender,
        'Occupation': selected_occupation,
        'product_type': selected_product_type,
        'product_detail': selected_product_detail,
        'Size': selected_size
    }])

    # Apply one-hot encoding to the categorical inputs
    # Need to create a dummy DataFrame with all possible columns from the original encoding
    dummy_data = pd.DataFrame(0, index=[0], columns=encoder.get_feature_names_out(
        ['store_location', 'product_category', 'product_type', 'product_detail', 'Size', 'Gender', 'Occupation', 'Season']
    ))

    # Fill in the values for the selected categories in the dummy DataFrame
    if f'Season_{selected_season}' in dummy_data.columns:
        dummy_data[f'Season_{selected_season}'] = 1
    if f'store_location_{selected_store_location}' in dummy_data.columns:
        dummy_data[f'store_location_{selected_store_location}'] = 1
    if f'product_type_{selected_product_type}' in dummy_data.columns:
         dummy_data[f'product_type_{selected_product_type}'] = 1
    if f'product_detail_{selected_product_detail}' in dummy_data.columns:
         dummy_data[f'product_detail_{selected_product_detail}'] = 1
    if f'Size_{selected_size}' in dummy_data.columns:
        dummy_data[f'Size_{selected_size}'] = 1
    if f'Gender_{selected_gender}' in dummy_data.columns:
        dummy_data[f'Gender_{selected_gender}'] = 1
    if f'Occupation_{selected_occupation}' in dummy_data.columns:
        dummy_data[f'Occupation_{selected_occupation}'] = 1


    # Define the list of all feature columns the model was trained on
    # This should match X_full columns
    all_model_features = [
        'transaction_qty', 'unit_price', 'Total_Bill', 'Age', 'Income',
        'store_location_Can Tho - Ninh Kieu', 'store_location_Da Nang - Hai Chau',
        'store_location_Da Nang - Thanh Khe', 'store_location_Hanoi - Ba Dinh',
        'store_location_Hanoi - Cau Giay', 'store_location_Hanoi - Hoan Kiem',
        'store_location_Ho Chi Minh City - District 1', 'store_location_Ho Chi Minh City - District 3',
        'store_location_Ho Chi Minh City - District 7', 'store_location_Ho Chi Minh City - Tan Binh',
        'store_location_Hue - City Center', 'store_location_Nha Trang - City Center',
        'product_type_Banh Mi', 'product_type_Cake', 'product_type_Coffee', 'product_type_Cookie', 'product_type_Freeze', 'product_type_Pastry', 'product_type_Tea',
        'product_detail_Americano', 'product_detail_Bac Xiu', 'product_detail_Banh Bong Lan Trung Muoi', 'product_detail_Banh Chuoi', 'product_detail_Banh Mi Ga Que', 'product_detail_Banh Mi Thit Nuong', 'product_detail_Banh Pate So', 'product_detail_Cappuccino', 'product_detail_Cookie Chocolate', 'product_detail_Cookie Oatmeal', 'product_detail_Espresso', 'product_detail_Freeze Ca Phe Phin Sua', 'product_detail_Freeze Chocolate', 'product_detail_Freeze Tra Xanh', 'product_detail_Latte', 'product_detail_Phin Den Da', 'product_detail_Phin Sua Da', 'product_detail_Tra Dao Cam Sa', 'product_detail_Tra Sen Vang',
        'Size_L', 'Size_M', 'Size_Regular', 'Size_S', 'Size_Slice',
        'Gender_Female', 'Gender_Male',
        'Occupation_Content creator', 'Occupation_Công nhân', 'Occupation_Doanh nhân', 'Occupation_Freelancer', 'Occupation_Giám đốc', 'Occupation_Giảng viên đại học', 'Occupation_Kinh doanh', 'Occupation_Kinh doanh online', 'Occupation_Nghề tự do', 'Occupation_Nhà đầu tư', 'Occupation_Nhân viên part-time', 'Occupation_Nhân viên văn phòng', 'Occupation_Quản lý', 'Occupation_Sinh viên, học sinh', 'Occupation_Thực tập sinh', 'Occupation_Trưởng phòng',
        'Season_Đông', 'Season_Hè', 'Season_Thu', 'Season_Xuân'
    ]


    # Combine the numerical inputs and the one-hot encoded categorical features
    # We need to ensure all columns that the model expects are present, even if their value is 0.
    # The dummy_data already has all potential encoded columns set to 0.
    # We need to carefully select and combine.

    # Start with the dummy_data which has all potential encoded columns set to 0
    final_input_df = dummy_data.copy()

    # Update the numerical columns from the input_df_numerical
    for col in input_df_numerical.columns:
        if col in final_input_df.columns:
             final_input_df[col] = input_df_numerical[col].iloc[0]
        else:
            # Add numerical columns if they are not already in dummy_data
            final_input_df[col] = input_df_numerical[col].iloc[0]


    # Ensure the columns in final_input_df match the columns the model was trained on (all_model_features)
    # This step is crucial to avoid shape mismatch errors during prediction
    missing_cols = set(all_model_features) - set(final_input_df.columns)
    for c in missing_cols:
        final_input_df[c] = 0
    # Ensure the order of columns is the same as the training data
    final_input_df = final_input_df[all_model_features]


    # Make prediction
    prediction_proba = model.predict_proba(final_input_df)

    # Get the class labels (integer labels from the model)
    class_labels_int = model.classes_

    # Map integer labels to original product category names using the encoder's categories
    product_category_mapping = {i: category for i, category in enumerate(product_categories_original)}

    # Find the predicted class with the highest probability
    predicted_class_index = np.argmax(prediction_proba)
    predicted_class_label_int = class_labels_int[predicted_class_index] # Get the integer label

    # Map the integer label back to the original product category name
    predicted_category = product_category_mapping.get(predicted_class_label_int, "Unknown Category")

    confidence = prediction_proba[0, predicted_class_index]


    # Display the prediction
    st.subheader("Kết quả dự báo:")
    st.write(f"Loại sản phẩm dự báo: {predicted_category}")
    st.write(f"Độ tin cậy: {confidence:.2f}")
