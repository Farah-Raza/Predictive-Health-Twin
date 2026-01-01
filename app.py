"""
Predictive Health Twin - Demo Application
A Python-based demo for chronic disease prediction using AI

Installation:
pip install streamlit pandas numpy scikit-learn plotly matplotlib seaborn

Run:
streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import time

# Page configuration
st.set_page_config(
    page_title="Predictive Health Twin",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with attractive styling
st.markdown("""
<style>
    /* Main styling */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Stylish header with gradient and shadow */
    .main-header {
        font-size: 4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: fadeInDown 1s ease-in;
    }
    
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #555;
        margin-bottom: 0.5rem;
        font-weight: 300;
    }
    
    .team-badge {
        text-align: center;
        font-size: 1.1rem;
        color: #764ba2;
        font-weight: 600;
        margin-bottom: 2rem;
        padding: 10px;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 25px;
        border: 2px solid #667eea;
        display: inline-block;
        width: 100%;
    }
    
    /* Animated gradient borders for cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border: 3px solid transparent;
        background-clip: padding-box;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        position: relative;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3);
    }
    
    /* Risk level cards with glowing effects */
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        padding: 25px;
        border-radius: 20px;
        font-size: 1.3rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(255, 107, 107, 0.4);
        border: 3px solid #ff8787;
        animation: pulse 2s infinite;
        font-weight: 600;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #ffa502 0%, #ff6348 100%);
        color: white;
        padding: 25px;
        border-radius: 20px;
        font-size: 1.3rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(255, 165, 2, 0.4);
        border: 3px solid #ffb733;
        font-weight: 600;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #26de81 0%, #20bf6b 100%);
        color: white;
        padding: 25px;
        border-radius: 20px;
        font-size: 1.3rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(38, 222, 129, 0.4);
        border: 3px solid #4cd964;
        font-weight: 600;
    }
    
    /* Stylish nudge boxes */
    .nudge-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-left: 6px solid #667eea;
        padding: 20px;
        margin: 15px 0;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        font-size: 1.05rem;
    }
    
    .nudge-box:hover {
        transform: translateX(10px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
    }
    
    /* Welcome card with beautiful borders */
    .welcome-card {
        background: white;
        padding: 40px;
        border-radius: 25px;
        border: 3px solid transparent;
        background-image: 
            linear-gradient(white, white),
            linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-origin: border-box;
        background-clip: padding-box, border-box;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
        margin: 20px 0;
    }
    
    .welcome-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .feature-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 15px 0;
        transition: all 0.3s ease;
    }
    
    .feature-box:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        transform: translateX(5px);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-right: 15px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 15px 30px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            box-shadow: 0 10px 40px rgba(255, 107, 107, 0.4);
        }
        50% {
            box-shadow: 0 10px 60px rgba(255, 107, 107, 0.6);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .slide-in {
        animation: slideInLeft 0.8s ease-out;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 15px;
        padding: 15px 25px;
        font-weight: 600;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        border: 2px solid #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Info box styling */
    .stAlert {
        border-radius: 15px;
        border-left: 6px solid #667eea;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'current_day' not in st.session_state:
    st.session_state.current_day = 0
if 'user_baseline' not in st.session_state:
    st.session_state.user_baseline = None

# Data Generation Functions
def generate_synthetic_patient_data(days=30, patient_type='diabetic'):
    """Generate synthetic health data for demo purposes"""
    np.random.seed(42)
    dates = [datetime.now() - timedelta(days=x) for x in range(days, 0, -1)]
    
    if patient_type == 'diabetic':
        base_hr = 72
        base_sleep = 7.5
        base_stress = 4
        base_steps = 7000
        base_glucose = 110
        
        data = {
            'date': dates,
            'heart_rate': [base_hr + np.random.normal(0, 8) + (5 if i > 20 and i < 25 else 0) for i in range(days)],
            'sleep_hours': [max(4, base_sleep + np.random.normal(0, 1.2) - (2 if i > 20 and i < 25 else 0)) for i in range(days)],
            'stress_level': [min(10, max(1, base_stress + np.random.normal(0, 1.5) + (3 if i > 20 and i < 25 else 0))) for i in range(days)],
            'steps': [max(1000, base_steps + np.random.normal(0, 2000) - (3000 if i > 20 and i < 25 else 0)) for i in range(days)],
            'blood_glucose': [max(70, base_glucose + np.random.normal(0, 15) + (30 if i > 20 and i < 25 else 0)) for i in range(days)],
            'systolic_bp': [120 + np.random.normal(0, 10) for i in range(days)],
            'diastolic_bp': [80 + np.random.normal(0, 8) for i in range(days)],
        }
        
    elif patient_type == 'hypertensive':
        base_hr = 78
        base_sleep = 6.8
        base_stress = 6
        base_steps = 5000
        
        data = {
            'date': dates,
            'heart_rate': [base_hr + np.random.normal(0, 10) + (8 if i > 22 and i < 27 else 0) for i in range(days)],
            'sleep_hours': [max(4, base_sleep + np.random.normal(0, 1.5) - (1.5 if i > 22 and i < 27 else 0)) for i in range(days)],
            'stress_level': [min(10, max(1, base_stress + np.random.normal(0, 2) + (2.5 if i > 22 and i < 27 else 0))) for i in range(days)],
            'steps': [max(1000, base_steps + np.random.normal(0, 1500) - (2000 if i > 22 and i < 27 else 0)) for i in range(days)],
            'blood_glucose': [100 + np.random.normal(0, 10) for i in range(days)],
            'systolic_bp': [135 + np.random.normal(0, 12) + (20 if i > 22 and i < 27 else 0) for i in range(days)],
            'diastolic_bp': [88 + np.random.normal(0, 10) + (12 if i > 22 and i < 27 else 0) for i in range(days)],
        }
    
    df = pd.DataFrame(data)
    df['risk_event'] = 0
    if patient_type == 'diabetic':
        df.loc[23:25, 'risk_event'] = 1
    elif patient_type == 'hypertensive':
        df.loc[25:27, 'risk_event'] = 1
    
    return df

def calculate_baseline(df):
    """Calculate user's personal baseline from historical data"""
    baseline = {
        'heart_rate_mean': df['heart_rate'].mean(),
        'heart_rate_std': df['heart_rate'].std(),
        'sleep_mean': df['sleep_hours'].mean(),
        'sleep_std': df['sleep_hours'].std(),
        'stress_mean': df['stress_level'].mean(),
        'stress_std': df['stress_level'].std(),
        'steps_mean': df['steps'].mean(),
        'glucose_mean': df['blood_glucose'].mean(),
        'systolic_mean': df['systolic_bp'].mean(),
        'diastolic_mean': df['diastolic_bp'].mean(),
    }
    return baseline

def train_prediction_model(df):
    """Train a simple predictive model"""
    features = []
    labels = []
    
    for i in range(3, len(df)):
        feat = [
            df.iloc[i-3:i]['heart_rate'].mean(),
            df.iloc[i-3:i]['heart_rate'].std(),
            df.iloc[i-3:i]['sleep_hours'].mean(),
            df.iloc[i-3:i]['stress_level'].mean(),
            df.iloc[i-3:i]['steps'].mean(),
            df.iloc[i-3:i]['blood_glucose'].mean(),
            df.iloc[i-3:i]['systolic_bp'].mean(),
            df.iloc[i-1]['heart_rate'] - df.iloc[i-2]['heart_rate'],
            df.iloc[i-1]['sleep_hours'] - df.iloc[i-2]['sleep_hours'],
        ]
        features.append(feat)
        labels.append(df.iloc[i]['risk_event'])
    
    X = np.array(features)
    y = np.array(labels)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

def predict_risk(model, scaler, recent_data, baseline):
    """Predict risk based on recent data"""
    try:
        feat = [
            recent_data['heart_rate'].mean(),
            recent_data['heart_rate'].std(),
            recent_data['sleep_hours'].mean(),
            recent_data['stress_level'].mean(),
            recent_data['steps'].mean(),
            recent_data['blood_glucose'].mean(),
            recent_data['systolic_bp'].mean(),
            recent_data.iloc[-1]['heart_rate'] - recent_data.iloc[-2]['heart_rate'],
            recent_data.iloc[-1]['sleep_hours'] - recent_data.iloc[-2]['sleep_hours'],
        ]
        
        X = scaler.transform([feat])
        risk_proba = model.predict_proba(X)[0]
        
        if len(risk_proba) > 1:
            risk_prob = risk_proba[1]
        else:
            risk_prob = 0.0
        
        hr_deviation = (recent_data['heart_rate'].iloc[-1] - baseline['heart_rate_mean']) / baseline['heart_rate_std']
        sleep_deviation = (recent_data['sleep_hours'].iloc[-1] - baseline['sleep_mean']) / baseline['sleep_std']
        stress_deviation = (recent_data['stress_level'].iloc[-1] - baseline['stress_mean']) / baseline['stress_std']
        
        return risk_prob, {
            'hr_deviation': hr_deviation,
            'sleep_deviation': sleep_deviation,
            'stress_deviation': stress_deviation
        }
    except Exception as e:
        return 0.0, {
            'hr_deviation': 0.0,
            'sleep_deviation': 0.0,
            'stress_deviation': 0.0
        }

def generate_nudge(risk_level, deviations, patient_type):
    """Generate personalized micro-nudges"""
    nudges = []
    
    if risk_level == 'HIGH':
        if patient_type == 'diabetic':
            nudges.append("🚨 HIGH RISK ALERT: Your patterns suggest elevated risk of blood glucose instability in the next 6-12 hours.")
            if deviations['stress_deviation'] > 1:
                nudges.append("• Your stress levels are unusually high. Take 10 minutes for deep breathing or meditation NOW.")
            if deviations['sleep_deviation'] < -1:
                nudges.append("• Poor sleep detected. Avoid sugary foods today and check blood glucose more frequently.")
            if abs(deviations['hr_deviation']) > 1:
                nudges.append("• Heart rate elevated. Avoid intense exercise and stay hydrated.")
            nudges.append("• Consider contacting your healthcare provider if symptoms worsen.")
        elif patient_type == 'hypertensive':
            nudges.append("🚨 HIGH RISK ALERT: Your patterns suggest elevated risk of blood pressure spike in the next 6-12 hours.")
            if deviations['stress_deviation'] > 1:
                nudges.append("• Stress levels critical. Take a 15-minute walk in a quiet area.")
            if deviations['sleep_deviation'] < -1:
                nudges.append("• Sleep deprivation detected. Limit caffeine and sodium intake today.")
            nudges.append("• Monitor your blood pressure every 4 hours today.")
    
    elif risk_level == 'MEDIUM':
        nudges.append("⚠️ ELEVATED RISK: Some patterns are trending away from your baseline.")
        if deviations['sleep_deviation'] < -0.5:
            nudges.append("• Sleep quality declining. Aim for 8 hours tonight.")
        if deviations['stress_deviation'] > 0.5:
            nudges.append("• Stress building up. Schedule 20 minutes of relaxation before evening.")
        nudges.append("• Stay mindful of your diet and medication schedule.")
    
    else:
        nudges.append("✅ LOW RISK: You're doing great! Keep up your healthy habits.")
        nudges.append("• Continue your current routine.")
    
    return nudges

# Main Application
def main():
    # Header with animation
    st.markdown('<h1 class="main-header">🏥 Predictive Health Twin</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Personalized Chronic Disease Management</p>', unsafe_allow_html=True)
    st.markdown('<div class="team-badge">✨ Team: DATA SURGE | VIT Bhopal 🎓</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Demo Configuration")
        patient_type = st.selectbox("Select Patient Profile", ['diabetic', 'hypertensive'])
        
        if st.button("🔄 Generate New Patient Data", use_container_width=True):
            st.session_state.data_generated = True
            st.session_state.current_day = 0
            with st.spinner("✨ Generating synthetic patient data..."):
                time.sleep(1)
                st.success("✅ Patient data generated!")
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 💡 About This Demo")
        st.info("""
        This demo showcases the Predictive Health Twin:
        
        🔹 Learns your personal baseline
        🔹 Analyzes patterns in real-time
        🔹 Predicts health risks early
        🔹 Delivers actionable micro-nudges
        """)
        
        st.markdown("---")
        st.markdown("### 🏆 Innovation Highlights")
        st.success("""
        ⚡ **N-of-1 Approach**: Personalized to YOU
        
        🎯 **Predictive, Not Reactive**: Stops problems before they start
        
        🔗 **Patient-Doctor Loop**: Seamless communication
        """)
    
    # Main content
    if st.session_state.data_generated:
        df = generate_synthetic_patient_data(days=30, patient_type=patient_type)
        baseline = calculate_baseline(df[:20])
        st.session_state.user_baseline = baseline
        
        model, scaler = train_prediction_model(df[:25])
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Real-Time Dashboard", "🔮 Prediction Engine", "📈 Historical Analysis", "👨‍⚕️ Clinical Dashboard"])
        
        with tab1:
            st.markdown("## 📊 Real-Time Health Monitoring")
            
            current_day = st.slider("📅 Demo Timeline (Days)", 0, 29, st.session_state.current_day, 
                                   help="Move slider to simulate different days")
            st.session_state.current_day = current_day
            
            col1, col2, col3, col4 = st.columns(4)
            current_data = df.iloc[current_day]
            
            with col1:
                st.metric("💓 Heart Rate", f"{current_data['heart_rate']:.0f} bpm",
                         delta=f"{current_data['heart_rate'] - baseline['heart_rate_mean']:.1f}")
            with col2:
                st.metric("😴 Sleep", f"{current_data['sleep_hours']:.1f} hrs",
                         delta=f"{current_data['sleep_hours'] - baseline['sleep_mean']:.1f}")
            with col3:
                st.metric("😰 Stress Level", f"{current_data['stress_level']:.0f}/10",
                         delta=f"{current_data['stress_level'] - baseline['stress_mean']:.1f}")
            with col4:
                st.metric("👟 Steps", f"{current_data['steps']:.0f}",
                         delta=f"{current_data['steps'] - baseline['steps_mean']:.0f}")
            
            col1, col2 = st.columns(2)
            with col1:
                if patient_type == 'diabetic':
                    st.metric("🩸 Blood Glucose", f"{current_data['blood_glucose']:.0f} mg/dL",
                             delta=f"{current_data['blood_glucose'] - baseline['glucose_mean']:.1f}")
            with col2:
                st.metric("💉 Blood Pressure", 
                         f"{current_data['systolic_bp']:.0f}/{current_data['diastolic_bp']:.0f} mmHg")
        
        with tab2:
            st.markdown("## 🔮 Predictive Analysis & Micro-Nudges")
            
            if current_day >= 3:
                recent_data = df.iloc[max(0, current_day-3):current_day+1]
                risk_prob, deviations = predict_risk(model, scaler, recent_data, baseline)
                
                if risk_prob > 0.6:
                    risk_level = 'HIGH'
                    risk_color = 'risk-high'
                elif risk_prob > 0.3:
                    risk_level = 'MEDIUM'
                    risk_color = 'risk-medium'
                else:
                    risk_level = 'LOW'
                    risk_color = 'risk-low'
                
                st.markdown(f'<div class="{risk_color}">⚡ RISK LEVEL: {risk_level} ({risk_prob*100:.1f}% probability)</div>', 
                           unsafe_allow_html=True)
                
                st.markdown("### 📱 Your Personalized Micro-Nudges")
                
                nudges = generate_nudge(risk_level, deviations, patient_type)
                for nudge in nudges:
                    st.markdown(f'<div class="nudge-box">{nudge}</div>', unsafe_allow_html=True)
                
                st.markdown("### 📉 Deviation from Your Personal Baseline")
                deviation_df = pd.DataFrame({
                    'Metric': ['Heart Rate', 'Sleep Quality', 'Stress Level'],
                    'Deviation (σ)': [deviations['hr_deviation'], 
                                     deviations['sleep_deviation'], 
                                     deviations['stress_deviation']]
                })
                
                fig = px.bar(deviation_df, x='Metric', y='Deviation (σ)', 
                           color='Deviation (σ)',
                           color_continuous_scale=['green', 'yellow', 'red'],
                           title="How Different Are You From Your Normal Self?")
                fig.add_hline(y=1, line_dash="dash", line_color="red", 
                             annotation_text="Warning Threshold")
                fig.add_hline(y=-1, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("⏳ Learning Phase: Collecting baseline data. Predictions will be available after 3 days of monitoring.")
        
        with tab3:
            st.markdown("## 📈 Historical Trends & Patterns")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['date'], y=df['heart_rate'], 
                                    mode='lines', name='Heart Rate', line=dict(color='#667eea', width=3)))
            fig.add_trace(go.Scatter(x=df['date'], y=df['sleep_hours']*10, 
                                    mode='lines', name='Sleep Hours (×10)', line=dict(color='#764ba2', width=3)))
            fig.add_trace(go.Scatter(x=df['date'], y=df['stress_level']*7, 
                                    mode='lines', name='Stress Level (×7)', line=dict(color='#f093fb', width=3)))
            
            risk_days = df[df['risk_event'] == 1]
            if not risk_days.empty:
                fig.add_vrect(x0=risk_days['date'].min(), x1=risk_days['date'].max(),
                             fillcolor="red", opacity=0.2, line_width=0,
                             annotation_text="⚠️ Risk Period", annotation_position="top left")
            
            fig.update_layout(title="30-Day Health Trends", 
                            xaxis_title="Date", yaxis_title="Value",
                            hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 🔗 Health Metrics Correlation")
            corr_data = df[['heart_rate', 'sleep_hours', 'stress_level', 
                           'steps', 'blood_glucose', 'systolic_bp']].corr()
            fig_corr = px.imshow(corr_data, text_auto=True, aspect="auto",
                                color_continuous_scale='RdBu_r')
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with tab4:
            st.markdown("## 👨‍⚕️ Clinical Dashboard for Healthcare Providers")
            
            st.info("🏥 This view is what your doctor sees - a summary of your predictive insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Patient Summary")
                st.write(f"**Patient Type:** {patient_type.title()}")
                st.write(f"**Monitoring Period:** 30 days")
                st.write(f"**High-Risk Days:** {df['risk_event'].sum()}")
                st.write(f"**Average Compliance:** 94%")
                
                st.subheader("Key Metrics vs Baseline")
                summary_df = pd.DataFrame({
                    'Metric': ['Heart Rate', 'Sleep', 'Stress', 'Steps'],
                    'Current': [
                        f"{df.iloc[-1]['heart_rate']:.0f} bpm",
                        f"{df.iloc[-1]['sleep_hours']:.1f} hrs",
                        f"{df.iloc[-1]['stress_level']:.0f}/10",
                        f"{df.iloc[-1]['steps']:.0f}"
                    ],
                    'Baseline': [
                        f"{baseline['heart_rate_mean']:.0f} bpm",
                        f"{baseline['sleep_mean']:.1f} hrs",
                        f"{baseline['stress_mean']:.0f}/10",
                        f"{baseline['steps_mean']:.0f}"
                    ]
                })
                st.table(summary_df)
            
            with col2:
                st.subheader("Risk Prediction Timeline")
                
                # Calculate risk for each day
                risk_timeline = []
                for i in range(3, len(df)):
                    recent = df.iloc[max(0, i-3):i+1]
                    risk, _ = predict_risk(model, scaler, recent, baseline)
                    risk_timeline.append(risk)
                
                risk_df = pd.DataFrame({
                    'Date': df['date'].iloc[3:],
                    'Risk Probability': risk_timeline
                })
                
                fig_risk = px.line(risk_df, x='Date', y='Risk Probability',
                                  title="30-Day Risk Trajectory")
                fig_risk.add_hline(y=0.6, line_dash="dash", line_color="red",
                                  annotation_text="High Risk Threshold")
                fig_risk.add_hline(y=0.3, line_dash="dash", line_color="orange",
                                  annotation_text="Medium Risk")
                st.plotly_chart(fig_risk, use_container_width=True)
            
            st.subheader("📋 Clinical Recommendations")
            st.markdown("""
            **Based on predictive analysis:**
            - Patient shows pattern of elevated risk around days 23-25
            - Recommend: Increase monitoring frequency during high-stress periods
            - Suggest: Review medication dosage with patient
            - Action: Schedule follow-up appointment to discuss lifestyle modifications
            """)
    
    else:
        # Welcome screen
        st.markdown("---")
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown("""
            ### 👈 Get Started
            Click **"Generate New Patient Data"** in the sidebar to see the demo in action!
            
            #### What You'll See:
            - Real-time health monitoring
            - Predictive risk analysis
            - Personalized micro-nudges
            - Clinical insights for doctors
            """)

if __name__ == "__main__":
    main()