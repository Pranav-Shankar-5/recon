# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from streamlit_option_menu import option_menu
# import joblib
# from components.header import render_header
# from components.file_upload import render_file_upload
# from components.data_preview import render_data_preview
# from components.reconciliation_results import render_reconciliation_results
# from components.charts import render_charts
# from components.manual_review import render_manual_review
# from components.reports import render_reports
# from components.advanced_filters import render_advanced_filters
# from components.notifications import render_notifications
# from components.help import render_help
# from sklearn.preprocessing import StandardScaler, LabelEncoder


# # Set page config
# st.set_page_config(page_title="Nostro Bank Reconciliation System", layout="wide")

# # Render header
# render_header()

# # Load the model
# @st.cache_resource
# def load_model():
#     model = joblib.load('/path/to/your/recon_model.pkl')
#     return model

# model = load_model()


# def preprocess_data(internal_records, foreign_records, is_prediction=False):
#     internal_records = internal_records[['ACCOUNT_ID', 'AMOUNT', 'STMT_DATE', 'DIRECTION']]
#     foreign_records = foreign_records[['ACCOUNT_ID', 'AMOUNT', 'STMT_DATE', 'DIRECTION']]

#     internal_records['AMOUNT'] = internal_records['AMOUNT'].abs()
#     foreign_records['AMOUNT'] = foreign_records['AMOUNT'].abs()

#     internal_records['STMT_DATE'] = pd.to_datetime(internal_records['STMT_DATE'], errors='coerce')
#     foreign_records['STMT_DATE'] = pd.to_datetime(foreign_records['STMT_DATE'], errors='coerce')

#     internal_records = internal_records.dropna(subset=['STMT_DATE'])
#     foreign_records = foreign_records.dropna(subset=['STMT_DATE'])

#     merged_data = pd.merge(
#         internal_records,
#         foreign_records,
#         on=['ACCOUNT_ID', 'AMOUNT', 'STMT_DATE', 'DIRECTION'],
#         how='outer',
#         indicator=True
#     )

#     merged_data['match_found'] = (merged_data['_merge'] == 'both').astype(int)
#     merged_data = merged_data.drop(columns=['_merge'])

#     X = merged_data[['ACCOUNT_ID', 'AMOUNT', 'STMT_DATE', 'DIRECTION']]
#     y = merged_data['match_found']

#     scaler = StandardScaler()
#     X['AMOUNT'] = scaler.fit_transform(X[['AMOUNT']])

#     label_encoder = LabelEncoder()
#     X['ACCOUNT_ID'] = label_encoder.fit_transform(X['ACCOUNT_ID'])
#     X['STMT_DATE'] = (X['STMT_DATE'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
#     X['DIRECTION'] = label_encoder.fit_transform(X['DIRECTION'])

#     if is_prediction:
#         return X, merged_data
#     return X, y

# # Navigation
# selected = option_menu(
#     menu_title=None,
#     options=["Home", "Upload Data", "View Results", "Generate Report", "Help"],
#     icons=["house", "cloud-upload", "list-task", "file-earmark-text", "question-circle"],
#     menu_icon="cast",
#     default_index=0,
#     orientation="horizontal",
# )

# # **Maintain Uploaded Data Across Pages**
# if "internal_df" not in st.session_state:
#     st.session_state.internal_df = None
# if "global_df" not in st.session_state:
#     st.session_state.global_df = None

# # Main content based on navigation
# if selected == "Home":
#     st.write("Welcome to the Nostro Bank Reconciliation System")

# elif selected == "Upload Data":
#     internal_df, global_df = render_file_upload()

#     # Save uploaded data for use in other pages
#     if internal_df is not None and global_df is not None:
#         st.session_state.internal_df = internal_df
#         st.session_state.global_df = global_df

#         # Preprocess and predict using the loaded model
#         X_internal, merged_data_internal = preprocess_data(internal_df, global_df, is_prediction=True)
#         predictions = model.predict(X_internal)

#         # Add predictions to the data and display
#         merged_data_internal['Predicted_Match'] = predictions
#         st.write(merged_data_internal)

# elif selected == "View Results":
#     if st.session_state.internal_df is not None and st.session_state.global_df is not None:
#         render_data_preview()
#         render_reconciliation_results(st.session_state.internal_df, st.session_state.global_df)
#         render_charts()
#         render_manual_review()
#     else:
#         st.warning("Please upload both Internal and Global transaction files first.")

# elif selected == "Generate Report":
#     render_reports()

# elif selected == "Help":
#     render_help()

# # Render advanced filters and notifications
# render_advanced_filters()
# render_notifications()

# # Dark mode toggle
# if st.sidebar.checkbox("Dark Mode"):
#     st.markdown("""
#         <style>
#         .stApp {
#             background-color: #2b2b2b;
#             color: #ffffff;
#         }
#         </style>
#     """, unsafe_allow_html=True)

import os
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
from tqdm import tqdm
import io

# Set the path where the model will be saved and loaded
MODEL_PATH = os.path.join(os.getcwd(), 'recon_model.pkl')  # Use relative path

def load_data(internal_file, foreign_file):
    # Read the uploaded Excel files (which are in memory as BytesIO)
    internal_records = pd.read_excel(internal_file)
    foreign_records = pd.read_excel(foreign_file)
    return internal_records, foreign_records

def preprocess_data(internal_records, foreign_records, is_prediction=False):
    # Check if necessary columns are present
    required_columns = ['ACCOUNT_ID', 'AMOUNT', 'STMT_DATE', 'DIRECTION']
    missing_columns_internal = [col for col in required_columns if col not in internal_records.columns]
    missing_columns_foreign = [col for col in required_columns if col not in foreign_records.columns]

    if missing_columns_internal or missing_columns_foreign:
        st.error(f"Missing columns in internal records: {missing_columns_internal}")
        st.error(f"Missing columns in foreign records: {missing_columns_foreign}")
        return None, None

    internal_records = internal_records[required_columns]
    foreign_records = foreign_records[required_columns]

    # Preprocess the 'AMOUNT' and 'STMT_DATE' columns
    internal_records['AMOUNT'] = internal_records['AMOUNT'].abs()
    foreign_records['AMOUNT'] = foreign_records['AMOUNT'].abs()

    internal_records['STMT_DATE'] = pd.to_datetime(internal_records['STMT_DATE'], errors='coerce')
    foreign_records['STMT_DATE'] = pd.to_datetime(foreign_records['STMT_DATE'], errors='coerce')

    # Drop rows with invalid dates
    internal_records = internal_records.dropna(subset=['STMT_DATE'])
    foreign_records = foreign_records.dropna(subset=['STMT_DATE'])

    # Merge the records
    merged_data = pd.merge(
        internal_records,
        foreign_records,
        on=['ACCOUNT_ID', 'AMOUNT', 'STMT_DATE', 'DIRECTION'],
        how='outer',
        indicator=True
    )

    merged_data['match_found'] = (merged_data['_merge'] == 'both').astype(int)
    merged_data = merged_data.drop(columns=['_merge'])

    X = merged_data[['ACCOUNT_ID', 'AMOUNT', 'STMT_DATE', 'DIRECTION']]
    y = merged_data['match_found']

    # Normalize 'AMOUNT' column
    scaler = StandardScaler()
    X['AMOUNT'] = scaler.fit_transform(X[['AMOUNT']])

    # Label encoding for categorical columns
    label_encoder = LabelEncoder()
    X['ACCOUNT_ID'] = label_encoder.fit_transform(X['ACCOUNT_ID'])
    X['STMT_DATE'] = (X['STMT_DATE'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')
    X['DIRECTION'] = label_encoder.fit_transform(X['DIRECTION'])

    if is_prediction:
        return X, merged_data
    return X, y

def train_and_predict_model(internal_records, foreign_records, model=None, is_prediction=False):
    # Preprocess the data
    X, merged_data = preprocess_data(internal_records, foreign_records, is_prediction)
    if X is None:  # Handle error in preprocessing
        return None

    if not is_prediction:
        y = merged_data['match_found']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(random_state=42)
        for _ in tqdm(range(1), desc="Training Model"):
            model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        st.write("Classification Report:")
        st.text(classification_report(y_test, predictions))
        st.write("Accuracy:", accuracy_score(y_test, predictions))

        joblib.dump(model, MODEL_PATH)  # Save the model to the specified path
        st.success("Model trained and saved successfully.")

    else:
        merged_data['FLAG'] = model.predict(X)
        output_data = merged_data[['AMOUNT', 'ACCOUNT_ID', 'STMT_DATE', 'DIRECTION', 'FLAG']]
        return output_data

def main():
    st.title("Reconciling Records with ML")
    
    # Upload files
    uploaded_internal = st.file_uploader("Upload Internal Records Excel", type="xlsx")
    uploaded_foreign = st.file_uploader("Upload Foreign Records Excel", type="xlsx")

    if uploaded_internal and uploaded_foreign:
        internal_records, foreign_records = load_data(uploaded_internal, uploaded_foreign)
        
        # Option to generate output
        if st.button('Generate Output'):
            if not os.path.exists(MODEL_PATH):  # Check if the model exists
                train_and_predict_model(internal_records, foreign_records)
            
            model = joblib.load(MODEL_PATH)  # Load the model from the specified path
            output_data = train_and_predict_model(internal_records, foreign_records, model=model, is_prediction=True)

            if output_data is not None:  # Handle case when prediction fails
                # Save output to Excel
                output_excel = io.BytesIO()
                output_data.to_excel(output_excel, index=False)
                output_excel.seek(0)

                # Provide download button
                st.download_button(
                    label="Download Output Excel",
                    data=output_excel,
                    file_name="recon_output.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()
