import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import io

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'encoders' not in st.session_state:
    st.session_state.encoders = {}
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None

def create_sample_dataset():
    """Create a sample student performance dataset"""
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Parental_Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
        'Study_Hours_Per_Week': np.random.randint(5, 40, n_samples),
        'Attendance_Percentage': np.random.randint(60, 100, n_samples),
        'Previous_Grade': np.random.randint(50, 100, n_samples),
        'Extracurricular_Activities': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
        'Sleep_Hours': np.random.uniform(4, 10, n_samples).round(1),
        'Tutoring': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'Physical_Activity_Hours': np.random.uniform(0, 10, n_samples).round(1),
        'Internet_Access': np.random.choice(['Yes', 'No'], n_samples, p=[0.8, 0.2]),
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable with realistic relationships
    base_score = 50
    base_score += (df['Study_Hours_Per_Week'] * 0.8)
    base_score += (df['Attendance_Percentage'] * 0.3)
    base_score += (df['Previous_Grade'] * 0.2)
    base_score += (df['Sleep_Hours'] * 2)
    base_score += np.where(df['Tutoring'] == 'Yes', 5, 0)
    base_score += np.where(df['Internet_Access'] == 'Yes', 3, 0)
    base_score += np.where(df['Extracurricular_Activities'] == 'Yes', 2, 0)
    base_score += np.random.normal(0, 5, n_samples)
    
    df['Final_Grade'] = np.clip(base_score, 0, 100).round(2)
    
    return df

def preprocess_data(df, is_training=True):
    """Preprocess the data by encoding categorical variables"""
    df_processed = df.copy()
    
    # Get target column name
    target_col = st.session_state.target_col if st.session_state.target_col else 'Final_Grade'
    
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    if is_training:
        st.session_state.encoders = {}
        for col in categorical_cols:
            if col != target_col:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                st.session_state.encoders[col] = le
    else:
        for col in categorical_cols:
            if col in st.session_state.encoders:
                le = st.session_state.encoders[col]
                df_processed[col] = le.transform(df_processed[col])
    
    return df_processed

def train_model(df, model_type='Random Forest'):
    """Train the selected model"""
    # Get target column name
    target_col = st.session_state.target_col if st.session_state.target_col else 'Final_Grade'
    
    # Preprocess data
    df_processed = preprocess_data(df, is_training=True)
    
    # Split features and target
    X = df_processed.drop(target_col, axis=1)
    y = df_processed[target_col]
    
    st.session_state.feature_columns = X.columns.tolist()
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Select and train model
    if model_type == 'Random Forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    else:
        model = LinearRegression()
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'train': {
            'r2': r2_score(y_train, y_pred_train),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'mae': mean_absolute_error(y_train, y_pred_train)
        },
        'test': {
            'r2': r2_score(y_test, y_pred_test),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'mae': mean_absolute_error(y_test, y_pred_test)
        }
    }
    
    st.session_state.model = model
    
    return model, metrics, X_test, y_test, y_pred_test

# Main App
st.markdown('<div class="main-header">🎓 Student Performance Prediction System</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/student-male--v1.png", width=100)
    st.markdown("### Navigation")
    page = st.radio("Go to:", ["📊 Data Overview", "🤖 Train Model", "🔮 Make Predictions", "📈 Analytics"])
    
    st.markdown("---")
    st.markdown("### About")
    st.info("This app predicts student final grades based on various factors like study hours, attendance, and previous performance.")

# Page: Data Overview
if page == "📊 Data Overview":
    st.markdown('<div class="sub-header">📊 Data Overview & Upload</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Upload Your Data")
        uploaded_file = st.file_uploader("Upload student data (CSV)", type=['csv'])
        
        if uploaded_file is not None:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success("✅ Data uploaded successfully!")
        else:
            if st.button("📝 Use Sample Dataset"):
                st.session_state.data = create_sample_dataset()
                st.success("✅ Sample dataset loaded!")
    
    with col2:
        st.markdown("#### Expected Columns")
        st.markdown("""
        - Gender
        - Parental_Education
        - Study_Hours_Per_Week
        - Attendance_Percentage
        - Previous_Grade
        - Extracurricular_Activities
        - Sleep_Hours
        - Tutoring
        - Physical_Activity_Hours
        - Internet_Access
        - Final_Grade (target)
        """)
    
    if st.session_state.data is not None:
        # Detect target column name
        possible_target_names = ['Final_Grade', 'final_grade', 'Grade', 'grade', 'Score', 'score', 'FinalGrade', 'final_score']
        target_col = None
        
        for col_name in possible_target_names:
            if col_name in st.session_state.data.columns:
                target_col = col_name
                break
        
        if target_col is None:
            st.error("⚠️ Could not find grade column. Please ensure your CSV has a column named 'Final_Grade', 'Grade', or 'Score'")
            st.info(f"📋 Your columns are: {', '.join(st.session_state.data.columns.tolist())}")
        else:
            # Store target column name in session state
            st.session_state.target_col = target_col
            
            st.markdown("---")
            st.markdown("#### Dataset Preview")
            st.dataframe(st.session_state.data.head(10), use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Students", len(st.session_state.data))
            with col2:
                st.metric("Average Grade", f"{st.session_state.data[target_col].mean():.2f}")
            with col3:
                st.metric("Highest Grade", f"{st.session_state.data[target_col].max():.2f}")
            with col4:
                st.metric("Lowest Grade", f"{st.session_state.data[target_col].min():.2f}")
        
        st.markdown("#### Statistical Summary")
        st.dataframe(st.session_state.data.describe(), use_container_width=True)
        
        # Visualizations
        st.markdown("#### Data Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Grade Distribution
            fig = px.histogram(st.session_state.data, x=target_col, nbins=30,
                             title='Distribution of Final Grades',
                             labels={target_col: 'Final Grade', 'count': 'Number of Students'},
                             color_discrete_sequence=['#1f77b4'])
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Study Hours vs Grade
            fig = px.scatter(st.session_state.data, x='Study_Hours_Per_Week', y=target_col,
                           title='Study Hours vs Final Grade',
                           labels={'Study_Hours_Per_Week': 'Study Hours per Week', target_col: 'Final Grade'},
                           color=target_col, color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Attendance vs Grade
            fig = px.scatter(st.session_state.data, x='Attendance_Percentage', y=target_col,
                           title='Attendance vs Final Grade',
                           labels={'Attendance_Percentage': 'Attendance %', target_col: 'Final Grade'},
                           color=target_col, color_continuous_scale='Plasma')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Gender distribution
            gender_counts = st.session_state.data['Gender'].value_counts()
            fig = px.pie(values=gender_counts.values, names=gender_counts.index,
                        title='Gender Distribution',
                        color_discrete_sequence=['#ff7f0e', '#2ca02c'])
            st.plotly_chart(fig, use_container_width=True)

# Page: Train Model
elif page == "🤖 Train Model":
    st.markdown('<div class="sub-header">🤖 Model Training</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data in the 'Data Overview' section first!")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Model Configuration")
            model_type = st.selectbox("Select Model Type", 
                                     ['Random Forest', 'Gradient Boosting', 'Linear Regression'])
            
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training model... Please wait..."):
                    model, metrics, X_test, y_test, y_pred_test = train_model(st.session_state.data, model_type)
                
                st.success("✅ Model trained successfully!")
                
                st.markdown("#### Training Metrics")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Training Set**")
                    st.metric("R² Score", f"{metrics['train']['r2']:.4f}")
                    st.metric("RMSE", f"{metrics['train']['rmse']:.4f}")
                    st.metric("MAE", f"{metrics['train']['mae']:.4f}")
                
                with col_b:
                    st.markdown("**Test Set**")
                    st.metric("R² Score", f"{metrics['test']['r2']:.4f}")
                    st.metric("RMSE", f"{metrics['test']['rmse']:.4f}")
                    st.metric("MAE", f"{metrics['test']['mae']:.4f}")
        
        with col2:
            if st.session_state.model is not None:
                st.markdown("#### Model Performance Visualization")
                
                # Get target column name
                target_col = st.session_state.target_col if st.session_state.target_col else 'Final_Grade'
                
                # Get predictions for visualization
                df_processed = preprocess_data(st.session_state.data, is_training=False)
                X = df_processed.drop(target_col, axis=1)
                y = df_processed[target_col]
                y_pred = st.session_state.model.predict(X)
                
                # Actual vs Predicted
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y, y=y_pred, mode='markers',
                                       name='Predictions',
                                       marker=dict(color='#1f77b4', size=8, opacity=0.6)))
                fig.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()],
                                       mode='lines', name='Perfect Prediction',
                                       line=dict(color='red', dash='dash')))
                fig.update_layout(title='Actual vs Predicted Grades',
                                xaxis_title='Actual Grade',
                                yaxis_title='Predicted Grade',
                                height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature Importance (for tree-based models)
                if model_type in ['Random Forest', 'Gradient Boosting']:
                    st.markdown("#### Feature Importance")
                    feature_importance = pd.DataFrame({
                        'Feature': st.session_state.feature_columns,
                        'Importance': st.session_state.model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(feature_importance, x='Importance', y='Feature',
                               orientation='h', title='Feature Importance Ranking',
                               color='Importance', color_continuous_scale='Blues')
                    st.plotly_chart(fig, use_container_width=True)

# Page: Make Predictions
elif page == "🔮 Make Predictions":
    st.markdown('<div class="sub-header">🔮 Make Predictions</div>', unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train a model first in the 'Train Model' section!")
    else:
        st.markdown("#### Enter Student Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ['Male', 'Female'])
            parental_education = st.selectbox("Parental Education", 
                                             ['High School', 'Bachelor', 'Master', 'PhD'])
            study_hours = st.slider("Study Hours per Week", 5, 40, 20)
            attendance = st.slider("Attendance Percentage", 60, 100, 85)
        
        with col2:
            previous_grade = st.slider("Previous Grade", 50, 100, 75)
            extracurricular = st.selectbox("Extracurricular Activities", ['Yes', 'No'])
            sleep_hours = st.slider("Sleep Hours per Day", 4.0, 10.0, 7.0, 0.1)
            tutoring = st.selectbox("Tutoring", ['Yes', 'No'])
        
        with col3:
            physical_activity = st.slider("Physical Activity Hours per Week", 0.0, 10.0, 3.0, 0.1)
            internet_access = st.selectbox("Internet Access", ['Yes', 'No'])
        
        st.markdown("---")
        
        if st.button("🎯 Predict Final Grade", type="primary"):
            # Create input dataframe
            input_data = pd.DataFrame({
                'Gender': [gender],
                'Parental_Education': [parental_education],
                'Study_Hours_Per_Week': [study_hours],
                'Attendance_Percentage': [attendance],
                'Previous_Grade': [previous_grade],
                'Extracurricular_Activities': [extracurricular],
                'Sleep_Hours': [sleep_hours],
                'Tutoring': [tutoring],
                'Physical_Activity_Hours': [physical_activity],
                'Internet_Access': [internet_access]
            })
            
            # Preprocess and predict
            input_processed = preprocess_data(input_data, is_training=False)
            prediction = st.session_state.model.predict(input_processed)[0]
            
            # Display prediction
            st.markdown("### Prediction Result")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                # Determine grade category and color
                if prediction >= 90:
                    grade_category = "Excellent (A)"
                    color = "#2ecc71"
                elif prediction >= 80:
                    grade_category = "Very Good (B)"
                    color = "#3498db"
                elif prediction >= 70:
                    grade_category = "Good (C)"
                    color = "#f39c12"
                elif prediction >= 60:
                    grade_category = "Satisfactory (D)"
                    color = "#e67e22"
                else:
                    grade_category = "Needs Improvement (F)"
                    color = "#e74c3c"
                
                st.markdown(f"""
                <div style='text-align: center; padding: 2rem; background-color: {color}; 
                            border-radius: 1rem; color: white;'>
                    <h1 style='margin: 0; font-size: 4rem;'>{prediction:.1f}</h1>
                    <h3 style='margin: 0.5rem 0 0 0;'>{grade_category}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Predicted Grade"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 60], 'color': "lightgray"},
                            {'range': [60, 70], 'color': "lightyellow"},
                            {'range': [70, 80], 'color': "lightblue"},
                            {'range': [80, 90], 'color': "lightgreen"},
                            {'range': [90, 100], 'color': "lime"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

# Page: Analytics
elif page == "📈 Analytics":
    st.markdown('<div class="sub-header">📈 Advanced Analytics</div>', unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data in the 'Data Overview' section first!")
    else:
        df = st.session_state.data
        target_col = st.session_state.target_col if st.session_state.target_col else 'Final_Grade'
        
        # Correlation Heatmap
        st.markdown("#### Correlation Analysis")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       text_auto='.2f',
                       aspect='auto',
                       color_continuous_scale='RdBu_r',
                       title='Feature Correlation Heatmap')
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance by Category
        st.markdown("#### Performance Analysis by Categories")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # By Gender
            if 'Gender' in df.columns:
                fig = px.box(df, x='Gender', y=target_col,
                            title='Grade Distribution by Gender',
                            color='Gender',
                            color_discrete_sequence=['#ff7f0e', '#2ca02c'])
                st.plotly_chart(fig, use_container_width=True)
            
            # By Tutoring
            if 'Tutoring' in df.columns:
                fig = px.box(df, x='Tutoring', y=target_col,
                            title='Grade Distribution by Tutoring',
                            color='Tutoring',
                            color_discrete_sequence=['#d62728', '#9467bd'])
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # By Parental Education
            if 'Parental_Education' in df.columns:
                fig = px.box(df, x='Parental_Education', y=target_col,
                            title='Grade Distribution by Parental Education',
                            color='Parental_Education')
                st.plotly_chart(fig, use_container_width=True)
            
            # By Extracurricular
            if 'Extracurricular_Activities' in df.columns:
                fig = px.box(df, x='Extracurricular_Activities', y=target_col,
                            title='Grade Distribution by Extracurricular Activities',
                            color='Extracurricular_Activities',
                            color_discrete_sequence=['#8c564b', '#e377c2'])
                st.plotly_chart(fig, use_container_width=True)
        
        # Study patterns
        st.markdown("#### Study Pattern Analysis")
        
        # Create study hour bins
        if 'Study_Hours_Per_Week' in df.columns:
            df['Study_Hours_Category'] = pd.cut(df['Study_Hours_Per_Week'], 
                                                bins=[0, 15, 25, 40],
                                                labels=['Low (5-15h)', 'Medium (16-25h)', 'High (26-40h)'])
            
            avg_by_study = df.groupby('Study_Hours_Category')[target_col].mean().reset_index()
            
            fig = px.bar(avg_by_study, x='Study_Hours_Category', y=target_col,
                        title='Average Grade by Study Hours Category',
                        labels={'Study_Hours_Category': 'Study Hours Category', 
                               target_col: 'Average Final Grade'},
                        color=target_col,
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Student Performance Prediction System | Built with Streamlit & Scikit-learn</p>
    <p>💡 Tip: Train your model with quality data for better predictions!</p>
</div>
""", unsafe_allow_html=True)
