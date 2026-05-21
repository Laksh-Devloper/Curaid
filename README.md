# CURA SPHERE

> Minimal chaos. Maximum control.

## Overview

**CURA SPHERE** is a structured, modular system designed to organize workflows, reduce mental load, and automate repetitive logic. It’s built with clarity first and scalability in mind.

The goal is simple: fewer manual decisions, cleaner systems, better outcomes.

## Features

* Modular and scalable architecture
* Clean separation of concerns
* Automation-friendly design
* Easy to extend and iterate

## Tech Stack

* **Frontend:** HTML, CSS, JavaScript
* **Backend:** Python, Django
* **APIs:** AI/Automation-ready (Gemini / GPT compatible)
* **Database:** SQLite / PostgreSQL

## Getting Started

### Prerequisites

* Python 3.9+
* Git
* Virtual environment (recommended)

### Installation

```bash
git clone https://github.com/Laksh-Devloper/Cura Sphere/tree/main
cd Cura Sphere
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run Locally

```bash
python manage.py runserver
```

## Project Structure

```
cura-sphere/
│── core/
│── api/
│── templates/
│── static/
│── manage.py
│── requirements.txt
```

## Use Cases

* Personal productivity tools
* Workflow automation systems
* AI-assisted backend experiments
* Portfolio demonstration project

## Future Scope

* Authentication & user roles
* Dashboard and analytics
* Advanced AI integrations
* Production deployment

## Contributing

Fork the repo, create a feature branch, and submit a pull request. Keep it readable. Keep it sane.

## License

MIT License.

---

Built with discipline, curiosity, and a refusal to accept messy systems.

# 🏥 Cura Sphere - AI-Powered Cura Sphere

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Django](https://img.shields.io/badge/Django-4.x-green.svg)
![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-orange.svg)

**Cura Sphere** is an intelligent health companion application that leverages AI to provide personalized health insights, disease risk predictions, mental health support, and comprehensive wellness tracking. Built with Django and powered by Google's Gemini AI, Cura Sphere helps users take control of their health journey.

---

## 📋 Table of Contents

- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Core Functionalities](#-core-functionalities)
- [API Endpoints](#-api-endpoints)
- [Database Models](#-database-models)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### 🤖 AI-Powered Chat Assistant
- **Multi-Mode AI Interaction**: Switch between General Health, Symptoms Predictor, Mental Health, and Disease Predictor modes
- **Conversational Interface**: Natural language processing for health queries
- **Session Management**: Multiple chat sessions with automatic title generation
- **Context-Aware Responses**: AI remembers conversation context within sessions

### 🏥 Health Assessment & Analytics
- **Comprehensive Health Scoring**: Calculate overall health scores based on:
  - Fitness metrics (BMI, exercise frequency)
  - Nutrition quality
  - Lifestyle factors (smoking, alcohol consumption)
  - Risk factor analysis
- **Personalized Recommendations**: AI-generated actionable health advice
- **Visual Health Dashboard**: Interactive charts and metrics visualization
- **BMI Calculator**: Automatic BMI calculation with health range assessment

### 🔬 Disease Risk Prediction
- **Diabetes Prediction**: ML-based diabetes risk assessment using 8 key parameters
- **Heart Disease Prediction**: Cardiovascular risk evaluation with 13 clinical parameters
- **Pre-trained Models**: Utilizes scikit-learn models for accurate predictions
- **Detailed Health Insights**: Personalized recommendations based on predictions

### 🧠 Mental Health Support
- **Emotional Wellness Tracking**: AI-powered mental health conversations
- **Activity Suggestions**: Personalized mental wellness activities (meditation, journaling, etc.)
- **To-Do Integration**: Automatically add suggested activities to task list
- **Supportive Guidance**: Empathetic responses and coping strategies

### 📄 Health Report Analysis
- **Multi-Format Support**: Upload PDF, JPG, PNG, or TXT health reports
- **OCR Technology**: Extract text from image-based reports using Tesseract
- **AI Analysis**: Comprehensive report interpretation with key findings
- **Actionable Insights**: Health metrics assessment and recommendations

### ✅ Smart To-Do Management
- **AI-Suggested Tasks**: Cura Sphere recommends health activities based on conversations
- **Due Date Tracking**: Set and monitor task deadlines
- **Status Management**: Mark tasks as complete or pending
- **Integrated Workflow**: Seamlessly manage health goals within chat interface

### 🔐 User Authentication
- **Email/Password Authentication**: Secure traditional login system
- **Google OAuth Integration**: One-click Google Sign-In
- **Password Encryption**: Cryptography-based password protection
- **Email Verification**: Token-based email verification system
- **Profile Management**: Update email and password settings

---

## 🛠 Technology Stack

### Backend
- **Framework**: Django 4.x
- **Language**: Python 3.8+
- **AI/ML**: 
  - Google Gemini 2.5 Flash (Generative AI)
  - scikit-learn (Disease prediction models)
  - NumPy (Data processing)
- **Authentication**: 
  - Django Auth System
  - Google OAuth 2.0
  - Django REST Framework
- **Database**: SQLite (Development) / PostgreSQL (Production-ready)

### Frontend
- **Templates**: Django Template Engine
- **Styling**: Custom CSS with modern UI/UX
- **JavaScript**: Vanilla JS for dynamic interactions
- **Charts**: Chart.js for health metrics visualization

### Additional Libraries
- **File Processing**:
  - PyPDF2 (PDF text extraction)
  - Pillow (Image processing)
  - pytesseract (OCR for images)
- **Security**:
  - cryptography (Password encryption)
  - certifi (SSL certificate verification)
- **API Integration**:
  - google-generativeai (Gemini API)
  - google-auth (OAuth authentication)

---

## 📁 Project Structure

```
Curo Tweaks/
├── case_companion/          # Main Django project
│   ├── settings.py          # Project settings
│   ├── urls.py              # Root URL configuration
│   ├── wsgi.py              # WSGI configuration
│   ├── diabetes_model.sav   # Pre-trained diabetes ML model
│   ├── heart_model.sav      # Pre-trained heart disease ML model
│   └── static/              # Static files (CSS, JS, images)
│
├── accounts/                # User authentication app
│   ├── models.py            # CustomUser, Contact, EmailVerification
│   ├── views.py             # Login, signup, Google OAuth handlers
│   ├── urls.py              # Account-related routes
│   ├── forms.py             # User forms
│   └── admin.py             # Admin configurations
│
├── chat/                    # AI chat and health features app
│   ├── models.py            # ChatSession, ChatMessage, UserTodo
│   ├── views.py             # Chat logic, AI handlers, health scoring
│   ├── urls.py              # Chat-related routes
│   └── admin.py             # Admin configurations
│
├── sim/                     # Simulation/Sign-in app
│   ├── views.py             # Google login view
│   └── urls.py              # Sim routes
│
├── templates/               # HTML templates
│   ├── index.html           # Landing page
│   ├── login.html           # Login/Signup page
│   ├── chat.html            # Main chat interface
│   ├── health_dashboard.html # Health metrics dashboard
│   ├── profile.html         # User profile page
│   └── google_login.html    # Google OAuth page
│
├── db.sqlite3               # SQLite database
├── manage.py                # Django management script
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Tesseract OCR (for health report image processing)
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/cura-sphere.git
cd "Curo Tweaks"
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Install Tesseract OCR
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows
# Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
```

### Step 5: Configure Environment Variables
Create a `.env` file in the project root:
```env
SECRET_KEY=your-django-secret-key
DEBUG=True
GEMINI_API_KEY=your-gemini-api-key
GOOGLE_OAUTH_CLIENT_ID=your-google-oauth-client-id
ENCRYPTION_KEY=your-encryption-key
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password
```

### Step 6: Run Migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### Step 7: Create Superuser (Optional)
```bash
python manage.py createsuperuser
```

### Step 8: Run Development Server
```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000/` in your browser.

---

## ⚙️ Configuration

### Google OAuth Setup
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google+ API
4. Create OAuth 2.0 credentials
5. Add authorized redirect URIs:
   - `http://127.0.0.1:8000/auth-receiver/`
   - `http://localhost:8000/auth-receiver/`
6. Copy Client ID to `settings.py`

### Gemini API Setup
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create an API key
3. Add to your environment variables or `chat/views.py`

### Email Configuration
For Gmail:
1. Enable 2-Factor Authentication
2. Generate App Password
3. Update `EMAIL_HOST_USER` and `EMAIL_HOST_PASSWORD` in `settings.py`

---

## 💡 Usage

### 1. User Registration & Login
- Navigate to `/login/` or `/signup/`
- Register with email/password or use Google Sign-In
- Access your profile at `/profile/`

### 2. AI Chat Interface
- Go to `/chat/` after logging in
- Select AI mode from dropdown:
  - **General Health**: Ask any health-related questions
  - **Symptoms Predictor**: Describe symptoms for AI analysis
  - **Mental Health**: Get emotional support and wellness tips
  - **Disease Predictor**: Input clinical parameters for risk assessment

### 3. Health Dashboard
- Click "Generate Health Report" in chat
- Fill in health metrics (age, weight, height, lifestyle factors)
- View comprehensive health score with visualizations
- Get personalized AI recommendations

### 4. To-Do Management
- Add tasks: `Add todo: Meditate for 10 minutes`
- Mark complete: `Done todo: 1` or `Done todo: Meditate`
- View list: `List todos`
- Clear completed: `Clear todos`

### 5. Health Report Upload
- Click upload icon in chat
- Select PDF/image of health report
- Receive AI-powered analysis and recommendations

---

## 🔧 Core Functionalities

### AI Chat Modes

#### 1. General Health Mode
```python
# Example query
"What are the benefits of drinking water?"

# AI provides evidence-based health advice
```

#### 2. Symptoms Predictor
```python
# Example input
"age: 30, symptoms: fatigue, headache, fever"

# AI analyzes symptoms and suggests next steps
```

#### 3. Mental Health Support
```python
# Example query
"I'm feeling stressed and anxious"

# AI provides supportive advice and suggests activities
# Offers to add activities to your to-do list
```

#### 4. Disease Predictor

**Diabetes Prediction:**
```python
# Required parameters
pregnancies: 2
glucose: 120
blood_pressure: 70
skin_thickness: 20
insulin: 80
bmi: 25.5
diabetes_pedigree_function: 0.5
age: 30
```

**Heart Disease Prediction:**
```python
# Required parameters
age: 50
sex: 1 (1=male, 0=female)
cp: 2 (chest pain type)
trestbps: 130 (resting blood pressure)
chol: 250 (cholesterol)
fbs: 0 (fasting blood sugar)
restecg: 0 (resting ECG)
thalach: 150 (max heart rate)
exang: 0 (exercise induced angina)
oldpeak: 1.5
slope: 1
ca: 0
thal: 2
```

### Health Scoring Algorithm
The system calculates a comprehensive health score (0-100) based on:
- **Fitness (30%)**: BMI + Exercise frequency
- **Nutrition (25%)**: Diet quality
- **Lifestyle (25%)**: Smoking + Alcohol consumption
- **Risk Factors (20%)**: Age-related risks

---

## 🌐 API Endpoints

### Authentication
- `POST /signup/` - User registration
- `POST /login/` - User login
- `POST /logout/` - User logout
- `POST /auth-receiver/` - Google OAuth callback
- `GET /profile/` - User profile page

### Chat & AI
- `GET /chat/` - Main chat interface
- `POST /chat/` - Send message to AI
- `POST /chat/generate_health_report/` - Generate health dashboard
- `GET /chat/health-dashboard/` - View health dashboard
- `GET /chat/get-sessions/` - Get all chat sessions
- `POST /chat/new-session/` - Create new chat session
- `GET /chat/load-session/<uuid>/` - Load specific session
- `DELETE /chat/delete-session/<uuid>/` - Delete session

### To-Do Management
- `POST /chat/mark_todo_done/<id>/` - Mark task complete
- `POST /chat/clear_completed_todos/` - Clear completed tasks

---

## 🗄️ Database Models

### CustomUser
```python
- email (EmailField, unique)
- username (CharField)
- encrypted_password (BinaryField)
- is_staff (BooleanField)
- is_superuser (BooleanField)
```

### ChatSession
```python
- user (ForeignKey to CustomUser)
- session_id (UUIDField, unique)
- title (CharField)
- created_at (DateTimeField)
- updated_at (DateTimeField)
- is_active (BooleanField)
```

### ChatMessage
```python
- user (ForeignKey to CustomUser)
- session (ForeignKey to ChatSession)
- message (TextField)
- bot_response (TextField)
- timestamp (DateTimeField)
```

### UserTodo
```python
- user (ForeignKey to CustomUser)
- task_description (CharField)
- suggested_by_curo (BooleanField)
- created_at (DateTimeField)
- due_date (DateField)
- completed (BooleanField)
- completed_at (DateTimeField)
```

### EmailVerification
```python
- email (EmailField, unique)
- token (UUIDField)
- created_at (DateTimeField)
- expires_at (DateTimeField)
- is_verified (BooleanField)
```

---

## 🎯 Future Enhancements

Based on the current implementation, here are potential upgrade areas:

### Phase 1: Enhanced AI Capabilities
- [ ] Voice-to-text input for chat
- [ ] Multi-language support
- [ ] AI-powered meal planning
- [ ] Exercise routine generator
- [ ] Sleep quality tracking

### Phase 2: Advanced Health Features
- [ ] Wearable device integration (Fitbit, Apple Watch)
- [ ] Medication reminder system
- [ ] Appointment scheduling with doctors
- [ ] Lab results tracking over time
- [ ] Family health profiles

### Phase 3: Social & Community
- [ ] Health challenges and goals
- [ ] Community forums
- [ ] Share progress with friends
- [ ] Leaderboards for health achievements
- [ ] Expert Q&A sessions

### Phase 4: Technical Improvements
- [ ] Mobile app (React Native/Flutter)
- [ ] Real-time notifications (WebSockets)
- [ ] Advanced data visualization
- [ ] Export health reports as PDF
- [ ] Offline mode support
- [ ] PostgreSQL migration for production
- [ ] Docker containerization
- [ ] CI/CD pipeline setup

### Phase 5: Security & Compliance
- [ ] HIPAA compliance features
- [ ] End-to-end encryption for health data
- [ ] Two-factor authentication
- [ ] Audit logging
- [ ] GDPR compliance tools

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to all functions
- Write unit tests for new features

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Laksh** - *Initial work* - [Laksh-Devloper](https://github.com/Laksh-Devloper)

---

## 🙏 Acknowledgments

- Google Gemini AI for powering the conversational interface
- Django community for the robust framework
- scikit-learn for machine learning capabilities
- All contributors and testers

---

## 📧 Contact

For questions, suggestions, or support:
- Email: casecompanion07@gmail.com
- GitHub Issues: [Create an issue](https://github.com/Laksh-Devloper/Cura Sphere/issues)

---

## 📊 Project Status

**Current Version**: 1.0.0  
**Status**: Active Development  
**Last Updated**: December 2025

---

**Made with ❤️ for better health and wellness**
