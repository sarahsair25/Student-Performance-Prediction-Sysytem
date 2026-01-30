# 🎓 Student Performance Prediction App

A comprehensive Streamlit application that predicts student final grades based on various academic and lifestyle factors using machine learning.

## ✨ Features

### 1. **Data Overview** 📊
- Upload your own CSV dataset or use the built-in sample dataset
- View statistical summaries and data distributions
- Interactive visualizations:
  - Grade distribution histogram
  - Study hours vs grades scatter plot
  - Attendance vs grades analysis
  - Gender distribution pie chart
- Key metrics dashboard (total students, average/highest/lowest grades)

### 2. **Model Training** 🤖
- Three machine learning algorithms to choose from:
  - **Random Forest** (Best for complex patterns)
  - **Gradient Boosting** (High accuracy)
  - **Linear Regression** (Simple and interpretable)
- Performance metrics:
  - R² Score
  - Root Mean Square Error (RMSE)
  - Mean Absolute Error (MAE)
- Visualizations:
  - Actual vs Predicted grades scatter plot
  - Feature importance ranking (for tree-based models)

### 3. **Make Predictions** 🔮
- Interactive form to input student information:
  - Demographics (Gender)
  - Academic (Study hours, Attendance, Previous grade)
  - Support (Tutoring, Internet access, Parental education)
  - Lifestyle (Sleep hours, Physical activity, Extracurricular activities)
- Real-time prediction with grade categorization
- Visual gauge chart showing predicted performance
- Color-coded results (A through F)

### 4. **Advanced Analytics** 📈
- Correlation heatmap for all numeric features
- Performance analysis by categories:
  - Gender comparison
  - Tutoring impact
  - Parental education influence
  - Extracurricular activities effect
- Study pattern analysis with categorized study hours

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Application
```bash
streamlit run student_performance_app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

## 📁 Data Format

Your CSV file should include these columns:

| Column Name | Type | Description | Example |
|------------|------|-------------|---------|
| Gender | Text | Male/Female | Male |
| Parental_Education | Text | Education level | Bachelor |
| Study_Hours_Per_Week | Number | Weekly study hours | 25 |
| Attendance_Percentage | Number | Class attendance % | 85 |
| Previous_Grade | Number | Last grade (0-100) | 78 |
| Extracurricular_Activities | Text | Yes/No | Yes |
| Sleep_Hours | Number | Daily sleep hours | 7.5 |
| Tutoring | Text | Yes/No | No |
| Physical_Activity_Hours | Number | Weekly exercise | 3.5 |
| Internet_Access | Text | Yes/No | Yes |
| Final_Grade | Number | Target variable (0-100) | 82.5 |

## 🎯 How to Use

### Option 1: Use Sample Data
1. Navigate to "Data Overview"
2. Click "Use Sample Dataset"
3. Explore the automatically generated data

### Option 2: Upload Your Own Data
1. Prepare your CSV file with the required columns
2. Go to "Data Overview"
3. Click "Upload student data (CSV)"
4. Select your file

### Train a Model
1. Go to "Train Model"
2. Select your preferred algorithm
3. Click "Train Model"
4. Review the performance metrics and visualizations

### Make Predictions
1. Navigate to "Make Predictions"
2. Fill in the student information using the sliders and dropdowns
3. Click "Predict Final Grade"
4. View the predicted grade with visual feedback

### Analyze Data
1. Go to "Analytics"
2. Explore correlations between features
3. Compare performance across different student categories
4. Identify patterns and trends

## 🔍 Key Insights

The app helps identify which factors most influence student performance:

- **Study Hours**: Strong positive correlation with grades
- **Attendance**: Higher attendance typically leads to better grades
- **Previous Performance**: Past grades are good predictors of future success
- **Sleep**: Adequate rest improves academic performance
- **Support Systems**: Tutoring and internet access show positive impacts

## 🛠️ Technical Details

**Built with:**
- **Streamlit**: Interactive web interface
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning algorithms

**Machine Learning Pipeline:**
1. Data preprocessing with label encoding for categorical variables
2. Train-test split (80-20)
3. Model training with hyperparameter optimization
4. Performance evaluation on test set
5. Feature importance analysis

## 📊 Model Performance

Expected performance on sample data:
- **Random Forest**: R² ≈ 0.85-0.90, RMSE ≈ 5-7
- **Gradient Boosting**: R² ≈ 0.83-0.88, RMSE ≈ 6-8
- **Linear Regression**: R² ≈ 0.75-0.80, RMSE ≈ 8-10

## 💡 Tips for Best Results

1. **Data Quality**: Ensure your data is clean and complete
2. **Sample Size**: More data (500+ records) leads to better predictions
3. **Feature Selection**: Include all relevant columns for accurate predictions
4. **Model Choice**: Start with Random Forest for best overall performance
5. **Regular Updates**: Retrain your model with new data regularly

## 🤝 Customization

You can easily customize the app:

- **Add more features**: Modify the input form and dataset structure
- **Change model parameters**: Adjust hyperparameters in the `train_model()` function
- **Add new visualizations**: Use Plotly to create additional charts
- **Modify styling**: Update the custom CSS in the main code

## 🐛 Troubleshooting

**App won't start:**
- Check Python version (3.8+)
- Verify all dependencies are installed
- Try `pip install --upgrade streamlit`

**Predictions seem inaccurate:**
- Ensure training data quality
- Try different model types
- Check for missing or inconsistent data

**Upload fails:**
- Verify CSV format and column names
- Check file encoding (UTF-8 recommended)
- Ensure all required columns are present

## 📝 License

Free to use and modify for educational and commercial purposes.

## 🙏 Acknowledgments

This app demonstrates the power of machine learning in educational analytics and can help educators identify at-risk students early and provide targeted interventions.

---

**Happy Predicting! 🎓✨**
