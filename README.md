# 🎓 Student Performance Prediction System

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.31.0-FF4B4B.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-F7931E.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**An AI-powered web application that predicts student academic performance using machine learning**

[Demo](#-demo) • [Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Data Format](#-data-format)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

The **Student Performance Prediction System** is a comprehensive machine learning application designed to help educators, administrators, and researchers predict student academic outcomes based on various factors including:

- 📚 Study habits and time investment
- 📊 Historical academic performance
- 👨‍👩‍👧‍👦 Socioeconomic factors (parental education, internet access)
- 🏃‍♂️ Lifestyle factors (sleep, physical activity, extracurriculars)
- 🎯 Support systems (tutoring availability)

Built with **Streamlit** for an intuitive user interface and **scikit-learn** for robust machine learning capabilities, this tool enables early identification of at-risk students and data-driven interventions.

---

## 🎬 Demo

### Dashboard Preview
<img width="960" height="460" alt="DAshboard" src="https://github.com/user-attachments/assets/ead33cbd-c1a1-48eb-a67f-9fba1c457925" />


### Prediction Interface
![Prediction](https://github.com/user-attachments/assets/8bfb715f-dbd6-4218-af51-a5709704485a)

### Analytics Dashboard

![Analytics](https://github.com/user-attachments/assets/afcdfe61-1ad3-4d6d-bae8-80c3d5dbfbbe)





 _**Live Demo:** [Coming Soon. In part 2 of this project]

---

## ✨ Features

### 📊 **Data Management & Visualization**
- 📁 Upload custom CSV datasets or use built-in sample data
- 📈 Interactive data visualizations with Plotly
- 📉 Statistical summaries and distribution analysis
- 🔍 Correlation heatmaps and pattern detection

### 🤖 **Machine Learning Models**
- 🌲 **Random Forest Regressor** - Best for complex, non-linear patterns
- 🚀 **Gradient Boosting Regressor** - High accuracy with sequential learning
- 📏 **Linear Regression** - Simple, interpretable baseline model
- 📊 Comprehensive performance metrics (R², RMSE, MAE)
- 🎯 Feature importance analysis

### 🔮 **Prediction System**
- 🖱️ User-friendly input interface with sliders and dropdowns
- ⚡ Real-time grade prediction
- 🎨 Color-coded performance categories (A through F)
- 📊 Visual gauge charts for intuitive result interpretation

### 📈 **Advanced Analytics**
- 🔗 Feature correlation analysis
- 📦 Performance comparison by categories (gender, tutoring, activities)
- 📚 Study pattern analysis with automated binning
- 🎯 Identification of key success factors

### 🎨 **User Experience**
- 🌐 Responsive web interface
- 🎭 Clean, modern UI with custom styling
- 📱 Mobile-friendly design
- 🔄 Session state management for seamless workflow

---

## 🛠️ Technology Stack

### **Core Technologies**
- **[Python 3.8+](https://www.python.org/)** - Programming language
- **[Streamlit](https://streamlit.io/)** - Web application framework
- **[scikit-learn](https://scikit-learn.org/)** - Machine learning library
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation
- **[NumPy](https://numpy.org/)** - Numerical computing
- **[Plotly](https://plotly.com/)** - Interactive visualizations

### **Machine Learning Pipeline**
```
Data Input → Preprocessing → Feature Engineering → Model Training → Evaluation → Prediction
```

---

## 📥 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)


## 🚀 Usage

### Option 1: Using Sample Data

1. Launch the application
2. Navigate to **"📊 Data Overview"**
3. Click **"📝 Use Sample Dataset"**
4. Explore the pre-generated data (500 students)

### Option 2: Upload Your Own Data

1. Prepare your CSV file following the [data format](#-data-format)
2. Navigate to **"📊 Data Overview"**
3. Click **"Upload student data (CSV)"**
4. Select your file

### Training a Model

1. Go to **"🤖 Train Model"**
2. Select your preferred algorithm:
   - Random Forest (recommended for most cases)
   - Gradient Boosting (for higher accuracy)
   - Linear Regression (for interpretability)
3. Click **"🚀 Train Model"**
4. Review performance metrics and visualizations

### Making Predictions

1. Navigate to **"🔮 Make Predictions"**
2. Input student information:
   - **Demographics**: Gender
   - **Academic**: Study hours, attendance, previous grades
   - **Support**: Tutoring, internet access, parental education
   - **Lifestyle**: Sleep, physical activity, extracurriculars
3. Click **"🎯 Predict Final Grade"**
4. View results with color-coded grade categories

### Analyzing Data

1. Go to **"📈 Analytics"**
2. Explore:
   - Correlation heatmaps
   - Performance by demographics
   - Impact of study patterns
   - Influence of support systems

---

## 📁 Data Format

Your CSV file should include the following columns:

| Column Name | Data Type | Range/Values | Description |
|-------------|-----------|--------------|-------------|
| `Gender` | Categorical | Male, Female | Student gender |
| `Parental_Education` | Categorical | High School, Bachelor, Master, PhD | Highest parental education level |
| `Study_Hours_Per_Week` | Numeric | 5-40 | Weekly study hours |
| `Attendance_Percentage` | Numeric | 0-100 | Class attendance percentage |
| `Previous_Grade` | Numeric | 0-100 | Previous academic performance |
| `Extracurricular_Activities` | Categorical | Yes, No | Participation in activities |
| `Sleep_Hours` | Numeric | 4-10 | Average daily sleep hours |
| `Tutoring` | Categorical | Yes, No | Access to tutoring |
| `Physical_Activity_Hours` | Numeric | 0-10 | Weekly exercise hours |
| `Internet_Access` | Categorical | Yes, No | Home internet availability |
| `Final_Grade` | Numeric | 0-100 | **Target variable** - Final grade to predict |

### Sample CSV Structure
```csv
Gender,Parental_Education,Study_Hours_Per_Week,Attendance_Percentage,Previous_Grade,Extracurricular_Activities,Sleep_Hours,Tutoring,Physical_Activity_Hours,Internet_Access,Final_Grade
Male,Bachelor,25,85,78,Yes,7.5,No,3.5,Yes,82.5
Female,Master,30,92,85,Yes,8.0,Yes,4.0,Yes,88.3
Male,High School,15,75,65,No,6.5,No,2.0,No,68.7
```

---

## 📊 Model Performance

### Expected Performance Metrics

Based on the sample dataset of 500 students:

| Model | R² Score | RMSE | MAE | Training Time |
|-------|----------|------|-----|---------------|
| **Random Forest** | 0.85-0.90 | 5-7 | 4-6 | ~2-3 seconds |
| **Gradient Boosting** | 0.83-0.88 | 6-8 | 5-7 | ~3-5 seconds |
| **Linear Regression** | 0.75-0.80 | 8-10 | 6-8 | <1 second |

### Feature Importance (Random Forest)

Top 5 most influential features:
1. **Study Hours per Week** (25-30% importance)
2. **Attendance Percentage** (20-25% importance)
3. **Previous Grade** (15-20% importance)
4. **Sleep Hours** (10-15% importance)
5. **Tutoring** (8-12% importance)

---



Contributions are welcome! Here's how you can help:

### Ways to Contribute
- 🐛 Report bugs and issues
- 💡 Suggest new features
- 📝 Improve documentation
- 🔧 Submit pull requests

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Run tests (if available)
5. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
6. Push to the branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to functions
- Comment complex logic
- Keep functions focused and modular

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 [Sarah Sair]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions...
```

---

## 🎓 Use Cases

### For Educators
- 📚 Identify at-risk students early
- 🎯 Develop targeted intervention strategies
- 📊 Track performance trends over time
- 💡 Optimize resource allocation

### For Administrators
- 📈 Make data-driven policy decisions
- 💰 Justify budget for support programs
- 🏆 Benchmark institutional performance
- 🔍 Identify systemic issues

### For Researchers
- 🔬 Study factors affecting academic success
- 📊 Analyze educational interventions
- 📝 Publish findings on student performance
- 🌍 Compare cross-institutional data

### For Students & Parents
- 📚 Understand performance factors
- 🎯 Set realistic goals
- 💪 Identify areas for improvement
- 📈 Track progress over time

---

## 🔮 Future Enhancements

- [ ] 🌐 Multi-language support
- [ ] 📱 Progressive Web App (PWA) version
- [ ] 🔐 User authentication and data privacy
- [ ] 📊 Advanced ensemble methods (Stacking, Voting)
- [ ] 🤖 Deep learning models (Neural Networks)
- [ ] 📈 Time-series analysis for longitudinal data
- [ ] 🔔 Alert system for at-risk students
- [ ] 📄 PDF report generation
- [ ] 🔗 Integration with Learning Management Systems (LMS)
- [ ] ☁️ Cloud deployment (AWS, GCP, Azure)
- [ ] 📊 A/B testing framework for interventions
- [ ] 🎨 Customizable themes and branding

---

## 🐛 Known Issues

- Large datasets (>10,000 rows) may cause performance slowdown
- Internet Explorer is not supported (use Chrome, Firefox, or Edge)
- Some visualizations may not render properly on mobile devices

For bug reports, please [open an issue](https://github.com/sarahsair25/Student-Performance-Prediction-System/issues).

---

## 📚 Documentation

### Additional Resources
- 📖 [User Guide](docs/user_guide.md)
- 🔧 [API Reference](docs/api_reference.md)
- 🎓 [Model Explanation](docs/model_explanation.md)
- ❓ [FAQ](docs/faq.md)

### Academic References
- Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance.
- Romero, C., & Ventura, S. (2010). Educational data mining: A review of the state of the art.
- Kotsiantis, S. B. (2012). Use of machine learning techniques for educational proposes.

---

## 💬 Contact

**Project Maintainer:** Sarah Sair

- 📧 Email: sarahsair@gmail.com
- 
- 💼 LinkedIn:      https://www.linkedin.com/in/sarahsair/
- Project Link:** [https://github.com/sarahsair25/Student-Performance-Prediction-Sysytem](https://github.com/sarahsair25/Student-Performance-Prediction-Sysytem)

---

## 🙏 Acknowledgments

- Thanks to the [Streamlit](https://streamlit.io/) team for the amazing framework
- [scikit-learn](https://scikit-learn.org/) for comprehensive ML tools
- [Plotly](https://plotly.com/) for beautiful interactive visualizations
- All contributors who have helped improve this project
- The education community for inspiring this work

---


---

<div align="center">

**Made with ❤️ for Education**

[⬆ Back to Top](#-student-performance-prediction-system)

</div>

