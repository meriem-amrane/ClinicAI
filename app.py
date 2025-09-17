# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ClinicAI - No-show Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv("data\clinical_data.csv")
        return df, None
    except FileNotFoundError:
        return None, "Dataset file not found. Please ensure 'data\clinical_data.csv' exists."
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"

@st.cache_resource
def load_model():
    """Load and cache the model"""
    try:
        model = joblib.load("clinic_ai_model.pkl")
        return model, None
    except FileNotFoundError:
        return None, "Model file not found. Please ensure 'clinic_ai_model.pkl' exists."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def create_age_distribution_plot(df):
    """Create an interactive age distribution plot"""
    fig = px.histogram(
        df, 
        x="Age", 
        color="No-show",
        nbins=30,
        title="Age Distribution by No-show Status",
        labels={"Age": "Age", "count": "Count"},
        color_discrete_map={"No": "#2E8B57", "Yes": "#DC143C"}
    )
    fig.update_layout(
        xaxis_title="Age",
        yaxis_title="Count",
        legend_title="No-show Status"
    )
    return fig

def create_feature_importance_plot(df):
    """Create additional visualizations for better insights"""
    # No-show rate by day of week
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    if 'AppointmentWeekday' in df.columns:
        weekday_stats = df.groupby('AppointmentWeekday')['No-show'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        weekday_stats['Weekday'] = [weekday_names[i] for i in weekday_stats['AppointmentWeekday']]
        
        fig = px.bar(
            weekday_stats,
            x='Weekday',
            y='No-show',
            title='No-show Rate by Day of Week',
            labels={'No-show': 'No-show Rate (%)'}
        )
        return fig
    return None

def get_risk_level_color(probability):
    """Return color based on risk level"""
    if probability < 0.3:
        return "#28a745"  # Green
    elif probability < 0.6:
        return "#ffc107"  # Yellow
    else:
        return "#dc3545"  # Red

def get_risk_level_text(probability):
    """Return risk level text based on probability"""
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"

def main():
    # Header
    st.markdown('<h1 class="main-header">🏥 ClinicAI - No-show Predictor</h1>', unsafe_allow_html=True)
    
    # Load data and model
    df, data_error = load_data()
    model, model_error = load_model()
    
    # Error handling
    if data_error:
        st.error(data_error)
        st.stop()
    
    if model_error:
        st.error(model_error)
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📊 Dataset Overview", "📈 Analytics", "🔮 Prediction", "ℹ️ About"]
    )
    
    if page == "📊 Dataset Overview":
        show_dataset_overview(df)
    elif page == "📈 Analytics":
        show_analytics(df)
    elif page == "🔮 Prediction":
        show_prediction_interface(model, df)
    elif page == "ℹ️ About":
        show_about_page()

def show_dataset_overview(df):
    """Display dataset overview section"""
    st.header("📊 Dataset Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    
    with col2:
        no_show_rate = (df["No-show"] == "Yes").mean() * 100
        st.metric("No-show Rate", f"{no_show_rate:.1f}%")
    
    with col3:
        st.metric("Features", df.shape[1])
    
    with col4:
        avg_age = df["Age"].mean()
        st.metric("Average Age", f"{avg_age:.1f} years")
    
    st.markdown("---")
    
    # Dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Dataset info
    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Data Types:**")
        st.write(df.dtypes.to_frame(name='Type'))
    
    with col2:
        st.write("**Missing Values:**")
        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values")

def show_analytics(df):
    """Display analytics section"""
    st.header("📈 Analytics Dashboard")
    
    # No-show statistics
    st.subheader("No-show Distribution")
    no_show_counts = df["No-show"].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig_pie = px.pie(
            values=no_show_counts.values,
            names=no_show_counts.index,
            title="No-show Distribution",
            color_discrete_map={"No": "#2E8B57", "Yes": "#DC143C"}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar chart
        fig_bar = px.bar(
            x=no_show_counts.index,
            y=no_show_counts.values,
            title="No-show Counts",
            color=no_show_counts.index,
            color_discrete_map={"No": "#2E8B57", "Yes": "#DC143C"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    
    # Age distribution
    st.subheader("Age Distribution Analysis")
    fig_age = create_age_distribution_plot(df)
    st.plotly_chart(fig_age, use_container_width=True)
    
    # Additional insights
    if 'AppointmentWeekday' in df.columns:
        st.markdown("---")
        st.subheader("Weekly Pattern Analysis")
        fig_weekday = create_feature_importance_plot(df)
        if fig_weekday:
            st.plotly_chart(fig_weekday, use_container_width=True)
    
    # Correlation analysis
    st.markdown("---")
    st.subheader("Feature Correlations")
    
    # Select only numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

def show_prediction_interface(model, df):
    """Display prediction interface"""
    st.header("🔮 No-show Prediction")
    
    st.info("Fill in the patient information below to predict the likelihood of a no-show.")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Demographics")
        age = st.slider("Age", min_value=0, max_value=120, value=35, help="Patient's age in years")
        gender = st.selectbox("Gender", ["Female", "Male"], help="Patient's gender")
        
        st.subheader("Medical Conditions")
        scholarship = st.selectbox("Has Scholarship", ["No", "Yes"], help="Whether patient has scholarship")
        hypertension = st.selectbox("Hypertension", ["No", "Yes"], help="Does patient have hypertension?")
        diabetes = st.selectbox("Diabetes", ["No", "Yes"], help="Does patient have diabetes?")
    
    with col2:
        st.subheader("Additional Information")
        alcoholism = st.selectbox("Alcoholism", ["No", "Yes"], help="Does patient have alcoholism?")
        handcap = st.selectbox("Disability", ["No", "Yes"], help="Does patient have any disability?")
        sms_received = st.selectbox("SMS Reminder Sent", ["No", "Yes"], help="Was SMS reminder sent?")
        
        st.subheader("Appointment Details")
        days_waiting = st.slider("Days Between Scheduling and Appointment", 
                                min_value=0, max_value=365, value=7,
                                help="Number of days between scheduling and appointment")
        
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        appointment_weekday = st.selectbox("Appointment Day", weekdays, 
                                         help="Day of the week for the appointment")
    
    # Convert inputs to model format
    gender_num = 0 if gender == "Female" else 1
    scholarship_num = 1 if scholarship == "Yes" else 0
    hypertension_num = 1 if hypertension == "Yes" else 0
    diabetes_num = 1 if diabetes == "Yes" else 0
    alcoholism_num = 1 if alcoholism == "Yes" else 0
    handcap_num = 1 if handcap == "Yes" else 0
    sms_received_num = 1 if sms_received == "Yes" else 0
    appointment_weekday_num = weekdays.index(appointment_weekday)
    
    # Create prediction button
    if st.button("🔮 Predict No-show Probability", type="primary"):
        try:
            # Create dataframe for prediction
            new_patient = pd.DataFrame({
                "Gender": [gender_num],
                "Age": [age],
                "Scholarship": [scholarship_num],
                "Hipertension": [hypertension_num],  # Note: keeping original column name
                "Diabetes": [diabetes_num],
                "Alcoholism": [alcoholism_num],
                "Handcap": [handcap_num],
                "SMS_received": [sms_received_num],
                "DaysWaiting": [days_waiting],
                "AppointmentWeekday": [appointment_weekday_num]
            })
            
            # Make prediction
            pred_prob = model.predict_proba(new_patient)[0][1]
            risk_level = get_risk_level_text(pred_prob)
            color = get_risk_level_color(pred_prob)
            
            # Display results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            # Probability gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pred_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "No-show Probability (%)"},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': color},
                       'steps': [
                           {'range': [0, 30], 'color': "lightgreen"},
                           {'range': [30, 60], 'color': "yellow"},
                           {'range': [60, 100], 'color': "lightcoral"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75, 'value': 50}}))
            
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Risk assessment
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("No-show Probability", f"{pred_prob*100:.1f}%")
            
            with col2:
                st.metric("Risk Level", risk_level)
            
            # Recommendations
            st.subheader("Recommendations")
            
            if pred_prob < 0.3:
                st.success("✅ **Low Risk**: Patient is likely to attend the appointment.")
                st.write("- Standard appointment confirmation is sufficient")
                st.write("- No additional follow-up needed")
            elif pred_prob < 0.6:
                st.warning("⚠️ **Medium Risk**: Consider additional follow-up.")
                st.write("- Send reminder SMS 24-48 hours before appointment")
                st.write("- Consider calling the patient to confirm")
                st.write("- Offer flexible scheduling options")
            else:
                st.error("🚨 **High Risk**: Strong likelihood of no-show.")
                st.write("- Multiple reminder contacts recommended")
                st.write("- Consider rescheduling to reduce waiting time")
                st.write("- Offer telehealth options if available")
                st.write("- Have backup patients ready for this slot")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write("Please check that all inputs are valid and the model is properly loaded.")

def show_about_page():
    """Display about page"""
    st.header("ℹ️ About ClinicAI")
    
    st.markdown("""
    ### Overview
    ClinicAI is a machine learning-powered application designed to predict patient no-shows for medical appointments. 
    By analyzing various patient and appointment factors, it helps healthcare providers optimize their scheduling and 
    reduce the impact of missed appointments.
    
    ### Features
    - **Dataset Analysis**: Comprehensive overview of historical appointment data
    - **Interactive Analytics**: Visual insights into no-show patterns and trends  
    - **Real-time Predictions**: Instant no-show probability calculation for new patients
    - **Risk Assessment**: Color-coded risk levels with actionable recommendations
    
    ### How It Works
    The system uses a trained machine learning model that considers multiple factors:
    
    **Patient Demographics:**
    - Age and gender
    - Scholarship status
    
    **Medical History:**
    - Chronic conditions (hypertension, diabetes)
    - Alcoholism
    - Disability status
    
    **Appointment Factors:**
    - Days between scheduling and appointment
    - Day of the week
    - SMS reminder status
    
    ### Model Performance
    The prediction model has been trained on historical appointment data and validated for accuracy. 
    Risk levels are categorized as:
    - **Low Risk** (0-30%): Patient likely to attend
    - **Medium Risk** (30-60%): Consider additional follow-up
    - **High Risk** (60%+): Strong likelihood of no-show
    
    ### Recommendations
    Based on the predicted risk level, the system provides specific recommendations for healthcare staff 
    to improve appointment attendance rates and optimize clinic operations.
    
    ---
    
    **Developed with ❤️ for healthcare optimization**
    """)

if __name__ == "__main__":
    main()