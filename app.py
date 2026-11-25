
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="SmartAQI Dashboard",
    page_icon="🌍",
    layout="wide"
)

# =====================================================
# FILE PATH
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "smart_aqi_dataset.csv"

# =====================================================
# DARK MODE DESIGN
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #080F1C;
    color: #F1F5F9;
}

header[data-testid="stHeader"] {
    background-color: #080F1C;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3, p, label {
    color: #F1F5F9;
}

.main-title {
    font-size: 40px;
    font-weight: 800;
    color: #38BDF8;
}

.subtitle {
    color: #94A3B8;
    font-size: 17px;
}

div[data-testid="stMetric"] {
    background: #111D30;
    border: 1px solid #263B56;
    padding: 20px;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8;
}

div[data-testid="stMetricValue"] {
    color: #38BDF8;
}

section[data-testid="stSidebar"] {
    background-color: #0D1929;
    border-right: 1px solid #263B56;
}

section[data-testid="stSidebar"] * {
    color: #E2E8F0;
}

.stButton > button {
    background-color: #0284C7;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 22px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #0369A1;
    color: white;
}

hr {
    border-color: #263B56;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATASET
# =====================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


if not DATA_PATH.is_file():
    st.error("Dataset not found! Run generate_data.py first.")
    st.stop()

try:
    data = load_data()
except Exception as e:
    st.error(f"Dataset loading error: {e}")
    st.stop()


required_columns = [
    "Traffic_Density",
    "PM25",
    "PM10",
    "Noise_Level",
    "Temperature",
    "Humidity",
    "AQI"
]

if not all(col in data.columns for col in required_columns):
    st.error("Dataset columns are missing. Regenerate the dataset.")
    st.stop()

if data.empty or data[required_columns].isnull().any().any():
    st.error("Dataset is empty or contains missing values.")
    st.stop()


# =====================================================
# FEATURE ENGINEERING FUNCTION
# =====================================================

def create_features(df):

    engineered = df.copy()

    # Interaction features
    engineered["Traffic_PM25"] = (
        engineered["Traffic_Density"] * engineered["PM25"]
    )

    engineered["PM_Ratio"] = (
        engineered["PM25"] / (engineered["PM10"] + 1)
    )

    engineered["Traffic_Noise"] = (
        engineered["Traffic_Density"] * engineered["Noise_Level"]
    )

    # Environmental interaction
    engineered["Temp_Humidity"] = (
        engineered["Temperature"] * engineered["Humidity"]
    )

    engineered["Heat_Index_Feature"] = (
        engineered["Temperature"] * (100 - engineered["Humidity"]) / 100
    )

    return engineered


# Create engineered dataset
engineered_data = create_features(data)


# =====================================================
# MACHINE LEARNING MODEL
# =====================================================

FEATURES = [
    "Traffic_Density",
    "PM25",
    "PM10",
    "Noise_Level",
    "Temperature",
    "Humidity",
    "Traffic_PM25",
    "PM_Ratio",
    "Traffic_Noise",
    "Temp_Humidity",
    "Heat_Index_Feature"
]


@st.cache_resource
def train_model():

    X = engineered_data[FEATURES]
    y = engineered_data["AQI"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    r2 = r2_score(y_test, predictions)

    return model, mae, rmse, r2, X_test, y_test, predictions


model, mae, rmse, r2, X_test, y_test, predictions = train_model()


# =====================================================
# AQI CATEGORY FUNCTION
# =====================================================

def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good", "🟢"

    elif aqi <= 100:
        return "Satisfactory", "🟡"

    elif aqi <= 200:
        return "Moderate", "🟠"

    elif aqi <= 300:
        return "Poor", "🔴"

    elif aqi <= 400:
        return "Very Poor", "🟣"

    else:
        return "Severe", "⚫"


# =====================================================
# PLOT STYLE
# =====================================================

def style_chart(fig):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111D30",
        plot_bgcolor="#111D30",
        font=dict(color="#E2E8F0"),
        title_font=dict(color="#38BDF8", size=20),
        margin=dict(l=30, r=30, t=60, b=30)
    )

    fig.update_xaxes(gridcolor="#263B56")
    fig.update_yaxes(gridcolor="#263B56")

    return fig


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.markdown("# 🌍 SmartAQI")
st.sidebar.caption("SMART TRAFFIC & AIR QUALITY")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Home",
        "AQI Prediction",
        "Feature Engineering",
        "Dataset Overview",
        "Traffic Analysis",
        "Environmental Analysis",
        "Model Performance",
        "About"
    ]
)

st.sidebar.divider()

st.sidebar.success("● MODEL TRAINED")

st.sidebar.caption(
    f"Dataset Records: {len(data)}"
)

st.sidebar.caption(
    "Algorithm: Random Forest Regression"
)


# =====================================================
# 1. HOME PAGE
# =====================================================

if page == "Home":

    st.markdown(
        '<div class="main-title">🌍 SmartAQI Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Smart Urban Traffic & Environmental Monitoring System'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.info(
        "Monitor traffic, explore environmental patterns "
        "and predict air quality using Machine Learning."
    )

    st.subheader("📊 Dashboard Statistics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", len(data))

    c2.metric(
        "Average AQI",
        f"{data['AQI'].mean():.2f}"
    )

    c3.metric(
        "Average Traffic",
        f"{data['Traffic_Density'].mean():.0f}"
    )

    c4.metric(
        "Average PM2.5",
        f"{data['PM25'].mean():.2f}"
    )

    st.divider()

    st.subheader("📈 AQI Distribution")

    fig = px.histogram(
        data,
        x="AQI",
        nbins=30,
        color_discrete_sequence=["#38BDF8"],
        title="Distribution of AQI Values"
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    st.subheader("🚗 Traffic vs AQI")

    fig2 = px.scatter(
        data,
        x="Traffic_Density",
        y="AQI",
        color="PM25",
        color_continuous_scale="Turbo",
        title="Traffic Density vs AQI"
    )

    st.plotly_chart(
        style_chart(fig2),
        use_container_width=True
    )


# =====================================================
# 2. AQI PREDICTION PAGE
# =====================================================

elif page == "AQI Prediction":

    st.title("🤖 AQI Prediction System")

    st.markdown(
        "Enter traffic and environmental parameters "
        "to predict the Air Quality Index."
    )

    st.info(
        "The Random Forest model uses the generated dataset "
        "and engineered features to estimate AQI."
    )

    st.divider()

    st.subheader("🚗 Traffic Information")

    col1, col2 = st.columns(2)

    with col1:
        traffic_input = st.slider(
            "Traffic Density",
            min_value=100,
            max_value=1500,
            value=750,
            step=10
        )

    with col2:
        noise_input = st.slider(
            "Noise Level",
            min_value=35,
            max_value=120,
            value=65
        )

    st.divider()

    st.subheader("🌫️ Air Pollution Information")

    col3, col4 = st.columns(2)

    with col3:
        pm25_input = st.slider(
            "PM2.5 Concentration",
            min_value=5.0,
            max_value=200.0,
            value=50.0
        )

    with col4:
        pm10_input = st.slider(
            "PM10 Concentration",
            min_value=10.0,
            max_value=300.0,
            value=90.0
        )

    st.divider()

    st.subheader("🌡️ Environmental Information")

    col5, col6 = st.columns(2)

    with col5:
        temperature_input = st.slider(
            "Temperature (°C)",
            min_value=18.0,
            max_value=42.0,
            value=30.0
        )

    with col6:
        humidity_input = st.slider(
            "Humidity (%)",
            min_value=25.0,
            max_value=95.0,
            value=60.0
        )

    st.write("")

    if st.button(
        "🔍 Predict AQI",
        use_container_width=True
    ):

        # Create user input dataframe

        user_input = pd.DataFrame([{
            "Traffic_Density": traffic_input,
            "PM25": pm25_input,
            "PM10": pm10_input,
            "Noise_Level": noise_input,
            "Temperature": temperature_input,
            "Humidity": humidity_input
        }])

        # Apply same feature engineering

        user_features = create_features(user_input)

        # Predict AQI

        predicted_aqi = model.predict(
            user_features[FEATURES]
        )[0]

        predicted_aqi = float(np.clip(predicted_aqi, 0, 500))

        category, emoji = get_aqi_category(predicted_aqi)

        st.divider()

        st.subheader("🎯 AQI Prediction Result")

        result1, result2 = st.columns(2)

        result1.metric(
            "Predicted AQI",
            f"{predicted_aqi:.2f}"
        )

        result2.metric(
            "Air Quality Category",
            f"{emoji} {category}"
        )

        # AQI gauge

        gauge = px.bar(
            x=["Predicted AQI"],
            y=[predicted_aqi],
            range_y=[0, 500],
            title="Predicted AQI Level",
            color_discrete_sequence=["#38BDF8"]
        )

        st.plotly_chart(
            style_chart(gauge),
            use_container_width=True
        )

        st.success(
            f"Prediction completed! Estimated AQI: "
            f"{predicted_aqi:.2f} ({category})"
        )

        st.caption(
            "This is an educational model prediction, "
            "not an official AQI measurement or health advisory."
        )


# =====================================================
# 3. FEATURE ENGINEERING PAGE
# =====================================================

elif page == "Feature Engineering":

    st.title("⚙️ Feature Engineering Application")

    st.markdown(
        "Transform raw environmental data into useful "
        "machine learning features."
    )

    st.info(
        "Feature engineering creates new variables from "
        "existing data to help machine learning models "
        "identify patterns."
    )

    st.divider()

    st.subheader("📂 Original Dataset")

    st.dataframe(
        data.head(10),
        use_container_width=True,
        hide_index=True
    )

    st.write(
        f"Original Features: {len(data.columns) - 1}"
    )

    st.divider()

    st.subheader("🛠️ Select Feature Engineering Techniques")

    add_traffic_pm25 = st.checkbox(
        "Traffic × PM2.5 Interaction",
        value=True
    )

    add_pm_ratio = st.checkbox(
        "PM2.5 / PM10 Ratio",
        value=True
    )

    add_traffic_noise = st.checkbox(
        "Traffic × Noise Interaction",
        value=True
    )

    add_temp_humidity = st.checkbox(
        "Temperature × Humidity Interaction",
        value=True
    )

    add_heat_feature = st.checkbox(
        "Temperature and Humidity Derived Feature",
        value=True
    )

    # Start with original data

    selected_features = data.copy()

    if add_traffic_pm25:
        selected_features["Traffic_PM25"] = (
            data["Traffic_Density"] * data["PM25"]
        )

    if add_pm_ratio:
        selected_features["PM_Ratio"] = (
            data["PM25"] / (data["PM10"] + 1)
        )

    if add_traffic_noise:
        selected_features["Traffic_Noise"] = (
            data["Traffic_Density"] * data["Noise_Level"]
        )

    if add_temp_humidity:
        selected_features["Temp_Humidity"] = (
            data["Temperature"] * data["Humidity"]
        )

    if add_heat_feature:
        selected_features["Heat_Index_Feature"] = (
            data["Temperature"] *
            (100 - data["Humidity"]) / 100
        )

    st.divider()

    st.subheader("📊 Engineered Dataset Results")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Original Columns",
        len(data.columns)
    )

    c2.metric(
        "Engineered Columns",
        len(selected_features.columns)
    )

    c3.metric(
        "New Features",
        len(selected_features.columns) - len(data.columns)
    )

    st.dataframe(
        selected_features.head(50).round(2),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("📈 Feature Correlation")

    corr = selected_features.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Engineered Feature Correlation Heatmap"
    )

    st.plotly_chart(
        style_chart(fig),
        use_container_width=True
    )

    st.subheader("📥 Export Engineered Dataset")

    csv = selected_features.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Engineered Dataset",
        data=csv,
        file_name="smart_aqi_engineered_dataset.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.success(
        "Feature engineering completed successfully!"
    )


# =====================================================
# 4. DATASET OVERVIEW
# =====================================================

elif page == "Dataset Overview":

    st.title("📁 Dataset Overview")

    c1, c2 = st.columns(2)

    c1.metric("Total Rows", data.shape[0])
    c2.metric("Total Columns", data.shape[1])

    st.subheader("Dataset Records")

    st.dataframe(
        data.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        data.describe().round(2),
        use_container_width=True
    )

    st.subheader("Missing Values")

    st.dataframe(
        data.isnull().sum().rename("Missing Values"),
        use_container_width=True
    )

    st.download_button(
        "📥 Download Dataset",
        data=data.to_csv(index=False),
        file_name="smart_aqi_dataset.csv",
        mime="text/csv"
    )


# =====================================================
# 5. TRAFFIC ANALYSIS
# =====================================================

elif page == "Traffic Analysis":

    st.title("🚗 Traffic Analysis")

    fig1 = px.scatter(
        data,
        x="Traffic_Density",
        y="PM25",
        color="AQI",
        color_continuous_scale="Turbo",
        title="Traffic vs PM2.5"
    )

    st.plotly_chart(
        style_chart(fig1),
        use_container_width=True
    )

    fig2 = px.scatter(
        data,
        x="Traffic_Density",
        y="Noise_Level",
        color="AQI",
        title="Traffic vs Noise Level"
    )

    st.plotly_chart(
        style_chart(fig2),
        use_container_width=True
    )

    fig3 = px.histogram(
        data,
        x="Traffic_Density",
        nbins=25,
        title="Traffic Density Distribution",
        color_discrete_sequence=["#2DD4BF"]
    )

    st.plotly_chart(
        style_chart(fig3),
        use_container_width=True
    )


# =====================================================
# 6. ENVIRONMENTAL ANALYSIS
# =====================================================

elif page == "Environmental Analysis":

    st.title("🌱 Environmental Analysis")

    c1, c2, c3 = st.columns(3)

    c1.metric("Average PM2.5", f"{data['PM25'].mean():.2f}")
    c2.metric("Average PM10", f"{data['PM10'].mean():.2f}")
    c3.metric("Average Noise", f"{data['Noise_Level'].mean():.2f}")

    fig1 = px.scatter(
        data,
        x="PM25",
        y="PM10",
        color="AQI",
        title="PM2.5 vs PM10"
    )

    st.plotly_chart(
        style_chart(fig1),
        use_container_width=True
    )

    fig2 = px.scatter(
        data,
        x="Temperature",
        y="AQI",
        color="Humidity",
        title="Temperature vs AQI"
    )

    st.plotly_chart(
        style_chart(fig2),
        use_container_width=True
    )

    fig3 = px.imshow(
        data.corr(numeric_only=True),
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap"
    )

    st.plotly_chart(
        style_chart(fig3),
        use_container_width=True
    )


# =====================================================
# 7. MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.title("🧠 Machine Learning Model Performance")

    st.write(
        "The Random Forest Regression model is trained "
        "using 80% of the dataset and evaluated using "
        "the remaining 20%."
    )

    st.subheader("Model Evaluation")

    c1, c2, c3 = st.columns(3)

    c1.metric("MAE", f"{mae:.2f}")
    c2.metric("RMSE", f"{rmse:.2f}")
    c3.metric("R² Score", f"{r2:.4f}")

    st.divider()

    st.subheader("Actual vs Predicted AQI")

    comparison = pd.DataFrame({
        "Actual AQI": y_test.values,
        "Predicted AQI": predictions
    })

    fig1 = px.scatter(
        comparison,
        x="Actual AQI",
        y="Predicted AQI",
        title="Actual AQI vs Predicted AQI",
        color_discrete_sequence=["#38BDF8"]
    )

    fig1.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=500,
        y1=500,
        line=dict(color="#F97316", dash="dash")
    )

    st.plotly_chart(
        style_chart(fig1),
        use_container_width=True
    )

    st.subheader("Feature Importance")

    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    )

    fig2 = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance",
        color="Importance",
        color_continuous_scale="Viridis"
    )

    fig2.update_layout(
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(
        style_chart(fig2),
        use_container_width=True
    )

    st.dataframe(
        importance.round(4),
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# 8. ABOUT
# =====================================================

elif page == "About":

    st.title("🌍 About SmartAQI")

    st.markdown("""
    ### SmartAQI - Smart Urban Air Quality Monitoring

    SmartAQI is a Machine Learning-based mini-project
    designed to explore the relationship between
    traffic congestion and environmental pollution.

    ### Project Objectives

    - Analyze traffic density.
    - Study PM2.5 and PM10 pollution.
    - Explore environmental parameters.
    - Predict AQI using Machine Learning.
    - Apply feature engineering techniques.
    - Visualize environmental patterns.

    ### Technologies Used

    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Plotly
    - Scikit-learn
    - Random Forest Regression

    ### Machine Learning

    The project uses Random Forest Regression to
    estimate AQI values from traffic and environmental
    parameters.

    """)

    st.warning(
        "The dataset is artificially generated. "
        "Predictions are educational estimates, not "
        "real-time or officially measured AQI values."
    )

    st.success(
        "SmartAQI | Making Cities Cleaner and Smarter 🌍"
    )
