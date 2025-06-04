import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder # For type hinting

# --- Page Configuration ---
st.set_page_config(
    page_title="CyberGuard AI - Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Cybersecurity Theme ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Create a style.css file in the same directory
local_css("style.css") # You'll create this file in Step 3

# For inline styling if you don't want a separate CSS file (simpler for now)



# --- Load Artifacts (Cached for performance) ---
@st.cache_resource # For objects that are expensive to create (like models)
def load_keras_model(path):
    return load_model(path)

@st.cache_data # For data that doesn't change
def load_other_artifact(path):
    if path.endswith('.pkl'):
        return joblib.load(path)
    elif path.endswith('.json'):
        with open(path, 'r') as f:
            return json.load(f)
    return None

MODEL_PATH = '/Users/veeshal/A-Hybrid-CNN-LSTM-Approach-for-Intelligent-Cyber-Intrusion-Detection-System/cnnlstm_dos_model_best.keras' # Use the path from ModelCheckpoint
SCALER_PATH = 'scaler_dos.pkl'
OHE_PATH = 'ohe_dos.pkl'
RFE_FEATURES_PATH = 'rfe_selected_feature_names_dos.json'
ALL_SCALER_COLUMNS_PATH = 'all_scaler_columns_dos.json'
CATEGORICAL_COLUMNS_PATH = 'categorical_column_names.json'
FULL_KDD_COLUMNS_PATH = 'full_kdd_column_names.json'

try:
    model = load_keras_model(MODEL_PATH)
    scaler = load_other_artifact(SCALER_PATH)
    ohe = load_other_artifact(OHE_PATH)
    rfe_selected_features = load_other_artifact(RFE_FEATURES_PATH)
    all_scaler_columns = load_other_artifact(ALL_SCALER_COLUMNS_PATH)
    categorical_columns_original = load_other_artifact(CATEGORICAL_COLUMNS_PATH)
    kdd_columns_full_list = load_other_artifact(FULL_KDD_COLUMNS_PATH)
    artifacts_loaded = True
except Exception as e:
    st.error(f"Error loading artifacts: {e}. Please ensure all artifact files are present.")
    st.error(f"Expected model: {MODEL_PATH}")
    st.error(f"Expected scaler: {SCALER_PATH}")
    st.error(f"Expected OHE: {OHE_PATH}")
    st.error(f"Expected RFE features list: {RFE_FEATURES_PATH}")
    st.error(f"Expected scaler columns list: {ALL_SCALER_COLUMNS_PATH}")
    st.error(f"Expected categorical columns list: {CATEGORICAL_COLUMNS_PATH}")
    artifacts_loaded = False


# --- Preprocessing Function for New Data ---
def preprocess_input_data(df_input_raw: pd.DataFrame,
                          ohe_encoder: OneHotEncoder,
                          cat_cols_orig: list,
                          scaler_obj: StandardScaler,
                          cols_for_scaler: list,
                          rfe_cols: list) -> np.ndarray | None:
    try:
        df_processed = df_input_raw.copy()

        # 1. One-Hot Encode categorical features using the loaded OHE
        # Ensure the categorical columns exist in the input
        missing_cat_cols = [col for col in cat_cols_orig if col not in df_processed.columns]
        if missing_cat_cols:
            st.warning(f"Uploaded CSV is missing expected categorical columns: {', '.join(missing_cat_cols)}. Predictions might be inaccurate.")
            # Optionally, add them with a default value or handle as needed
            for col in missing_cat_cols:
                df_processed[col] = "unknown" # Or a common value from training like 'other' for service

        cat_features_ohe = ohe_encoder.transform(df_processed[cat_cols_orig])
        ohe_feature_names = ohe_encoder.get_feature_names_out(cat_cols_orig)
        df_ohe = pd.DataFrame(cat_features_ohe, columns=ohe_feature_names, index=df_processed.index)

        # Drop original categorical columns and join OHE features
        df_processed = df_processed.drop(columns=cat_cols_orig).join(df_ohe)

        # 2. Align columns with what the scaler expects (cols_for_scaler)
        # These are all columns AFTER OHE but BEFORE RFE
        df_aligned = df_processed.reindex(columns=cols_for_scaler, fill_value=0)

        # 3. Scale numerical features using the loaded scaler
        scaled_features = scaler_obj.transform(df_aligned)
        df_scaled = pd.DataFrame(scaled_features, columns=cols_for_scaler, index=df_aligned.index)

        # 4. Select RFE features
        df_rfe = df_scaled[rfe_cols]

        # 5. Reshape for CNN-LSTM
        reshaped_data = df_rfe.values.reshape(df_rfe.shape[0], df_rfe.shape[1], 1)
        return reshaped_data
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        st.error("Please ensure your CSV has the correct KDD column format, including raw categorical features (protocol_type, service, flag).")
        return None

# --- UI Layout ---
st.title("🛡️ CyberGuard AI: Network Intrusion Detection System")
st.markdown("Upload connection data (CSV format similar to KDD datasets) to detect potential DoS attacks.")

st.sidebar.header("⚙️ Controls & Info")
uploaded_file = st.sidebar.file_uploader("Upload KDD-like CSV File", type=["csv"])

if artifacts_loaded:
    st.sidebar.success("✅ Model & Artifacts Loaded Successfully!")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Details:")
    st.sidebar.markdown(f"- **Type:** CNN-LSTM for DoS Detection")
    st.sidebar.markdown(f"- **Input Features (after RFE):** {len(rfe_selected_features)}")
    # You could add more details here if saved, e.g., training accuracy
else:
    st.sidebar.error("⚠️ Model or artifacts failed to load. Please check paths and files.")

st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit & TensorFlow/Keras")
st.sidebar.markdown("Cybersecurity Theme Example")

if uploaded_file is not None and artifacts_loaded and kdd_columns_full_list is not None:
    try:
        input_df_raw = pd.read_csv(uploaded_file, header=None, dtype=str) # Read all as str initially to preserve data
        num_cols_read = len(input_df_raw.columns)

        assigned_columns = False
        if num_cols_read == 41:
            st.info("Detected 41 columns, assuming headerless KDD features.")
            input_df_raw.columns = kdd_columns_full_list[:-2]
            assigned_columns = True
        elif num_cols_read == 43:
            st.info("Detected 43 columns, assuming headerless KDD features including target.")
            input_df_raw.columns = kdd_columns_full_list
            # Drop target columns if they exist by these names
            if 'attack' in input_df_raw.columns and 'level' in input_df_raw.columns:
                input_df_raw = input_df_raw.drop(columns=['attack', 'level'])
            else: # If names don't match, assume first 41 are features
                st.warning("43 columns read, but 'attack'/'level' not named as such. Taking first 41 as features.")
                input_df_raw = input_df_raw.iloc[:, :41]
                input_df_raw.columns = kdd_columns_full_list[:-2]
            assigned_columns = True
        else:
            st.warning(f"Read {num_cols_read} columns. This doesn't match typical KDD feature counts (41 or 43). Attempting to read with header inference.")
            # Re-read, allowing pandas to infer header if it exists
            # Use a fresh BytesIO object to allow re-reading
            uploaded_file.seek(0) # Reset file pointer
            input_df_raw = pd.read_csv(uploaded_file, dtype=str) # Read all as str initially
            st.info(f"Re-read CSV with inferred headers. Columns found: {input_df_raw.columns.tolist()}")
            # At this point, if it's not 41/43, the preprocessing might fail if columns don't match `all_scaler_columns`

        # Convert to appropriate types after columns are set (example)
        # This is a simplified type conversion. A more robust one would be needed.
        # For now, the preprocessing function will handle most of this.
        # Example: df_test_orig.info() from training can guide type conversion.
        # For OHE, string types for categorical columns are fine.
        # For scaler, numerical columns need to be numeric.

        st.markdown("---")
        st.subheader("Uploaded Data Sample (First 5 rows):")
        st.dataframe(input_df_raw.head())


        if st.button("🚨 Analyze Network Traffic"):
            with st.spinner("🧠 Analyzing connections... Please wait."):
                processed_data = preprocess_input_data(
                    input_df_raw,
                    ohe,
                    categorical_columns_original,
                    scaler,
                    all_scaler_columns,
                    rfe_selected_features
                )

            if processed_data is not None:
                predictions_proba = model.predict(processed_data)
                predictions_binary = (predictions_proba > 0.5).astype(int)

                input_df_raw['Prediction_Score'] = predictions_proba.flatten()
                input_df_raw['Detected_Attack (1=DoS, 0=Normal)'] = predictions_binary.flatten()

                st.markdown("---")
                st.subheader("🔬 Prediction Results:")

                num_dos = np.sum(predictions_binary)
                num_normal = len(predictions_binary) - num_dos
                total_conn = len(predictions_binary)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Total Connections Analyzed", value=total_conn)
                with col2:
                    st.markdown(f"<div class='success-box'>🛡️ Normal Connections: {num_normal}</div>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"<div class='error-box'>⚠️ Detected DoS Attacks: {num_dos}</div>", unsafe_allow_html=True)


                st.dataframe(input_df_raw)

                # Provide download link for results
                @st.cache_data # Cache the data for download
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv_results = convert_df_to_csv(input_df_raw)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_results,
                    file_name="predictions_cyberguard_ai.csv",
                    mime="text/csv",
                )
            else:
                st.error("Could not process data for prediction. Check logs.")
    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty. Please upload a valid file.")
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        st.error("Please ensure the uploaded CSV is in the correct KDD-like format.")

elif not artifacts_loaded:
    st.warning("Model artifacts are not loaded. File processing is disabled.")