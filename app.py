import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained multi-class model and encoder
model = joblib.load('multiclass_logistic_regression_model.joblib')
encoder = joblib.load('onehot_encoder.joblib')

# Define the selected features used during training for the multi-class model
selected_features = [
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

# Get the original categories from the encoder
original_categories = encoder.categories_
store_locations = original_categories[0]
product_categories_original = original_categories[1] # Original product categories
product_types = original_categories[2]
product_details = original_categories[3]
sizes = original_categories[4]
genders = original_categories[5]
occupations = original_categories[6]
seasons = original_categories[7]


# Streamlit app title
st.title("Dự báo Product Category")

# User input for features
st.sidebar.header("Nhập thông tin khách hàng")

# Input fields based on selected_features, mapping back to original categories for categorical inputs
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
selected_product_type = st.sidebar.selectbox("Loại sản phẩm", product_types)
selected_product_detail = st.sidebar.selectbox("Chi tiết sản phẩm", product_details)
selected_size = st.sidebar.selectbox("Kích cỡ", sizes)
selected_gender = st.sidebar.selectbox("Giới tính", genders)
selected_occupation = st.sidebar.selectbox("Nghề nghiệp", occupations)


# Button to trigger prediction
if st.sidebar.button("Dự báo"):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

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


    # Select only the encoded columns that are in selected_features
    encoded_input_features = dummy_data[
        [col for col in dummy_data.columns if col in selected_features and col not in input_data.keys()]
    ]

    # Concatenate numerical and encoded categorical features
    final_input_df = pd.concat([input_df, encoded_input_features], axis=1)

    # Ensure the columns in the final_input_df match the columns in X_train
    # Add missing columns with a value of 0
    missing_cols = set(selected_features) - set(final_input_df.columns)
    for c in missing_cols:
        final_input_df[c] = 0
    # Ensure the order of columns is the same as X_train
    final_input_df = final_input_df[selected_features]

    # Make prediction
    prediction_proba = model.predict_proba(final_input_df)

    # Get the class labels (original product categories from the encoder)
    class_labels = model.classes_ # These are integer labels (0, 1, 2, 3, 4)
    product_category_mapping = {i: category for i, category in enumerate(product_categories_original)}


    # Find the predicted class with the highest probability
    predicted_class_index = np.argmax(prediction_proba)
    predicted_class_label_int = class_labels[predicted_class_index] # Get the integer label

    # Map the integer label back to the original product category name
    predicted_category = product_category_mapping.get(predicted_class_label_int, "Unknown Category")

    confidence = prediction_proba[0, predicted_class_index]


    # Display the prediction
    st.subheader("Kết quả dự báo:")
    st.write(f"Loại sản phẩm dự báo: {predicted_category}")
    st.write(f"Độ tin cậy: {confidence:.2f}")
