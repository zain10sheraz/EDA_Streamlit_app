import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Configuration
st.set_page_config(
    page_title="EDA Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Exploratory Data Analysis Interface")

# 2. Sidebar: Dataset Ingestion
st.sidebar.header("Dataset Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV File for Analysis", type=["csv"])

if uploaded_file is not None:
    # Read dataset
    df = pd.read_csv(uploaded_file)
    
    # 3. Dataset Overview
    st.subheader("Dataset Preview & Metadata")
    
    st.write("**First 5 Rows:**")
    st.dataframe(df.head())

    st.write(f"**Shape:** `{df.shape[0]}` rows, `{df.shape[1]}` columns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Column Data Types:**")
        st.dataframe(pd.DataFrame(df.dtypes, columns=["Data Type"]))

    with col2:
        st.write("**Missing Values per Column:**")
        missing_df = pd.DataFrame({
            'Missing Count': df.isnull().sum(),
            'Missing %': (df.isnull().sum() / len(df)) * 100
        })
        st.dataframe(missing_df)

    # Basic statistics for numerical columns
    st.write("**Basic Numerical Statistics:**")
    st.dataframe(df.describe().T[['mean', '50%', 'min', 'max']].rename(columns={'50%': 'median'}))

    # 4. Attribute Selection
    st.sidebar.header("Attribute Selection")
    selected_col = st.sidebar.selectbox("Select Attribute for Visualization", df.columns)

    # Detect column type
    if pd.api.types.is_numeric_dtype(df[selected_col]):
        column_type = "Numerical"
    else:
        column_type = "Categorical"

    # 5. Visualization Rendering
    st.subheader("Visualization")

    fig, ax = plt.subplots(figsize=(8, 4))

    if column_type == "Numerical":
        sns.histplot(df[selected_col].dropna(), kde=True, ax=ax, color="skyblue")
        ax.set_title(f"Histogram of {selected_col}")
        ax.set_ylabel("Frequency")
        ax.set_xlabel(selected_col)
        st.pyplot(fig)
    else:
        sns.countplot(data=df, x=selected_col, ax=ax, palette="Blues_d")
        ax.set_title(f"Bar Chart of {selected_col}")
        ax.set_ylabel("Count")
        ax.set_xlabel(selected_col)
        plt.xticks(rotation=45)
        st.pyplot(fig)

else:
    st.info("Please upload a CSV file to start EDA.")