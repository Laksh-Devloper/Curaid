import json
import pickle
from datetime import date, datetime, timedelta
import requests
import numpy as np
import google.generativeai as genai
from django.contrib.auth import login, logout
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
import PyPDF2  # Add to requirements: PyPDF2
from PIL import Image
import pytesseract  # For OCR on images
from accounts.models import CustomUser
from .models import ChatMessage, UserTodo ,ChatSession

# --- Configuration & Model Loading ---

# Load ML models
DIABETES_MODEL = pickle.load(open('case_companion/diabetes_model.sav', 'rb'))
HEART_MODEL = pickle.load(open('case_companion/heart_model.sav', 'rb'))

# Configure Gemini API
GEMINI_API_KEY = ""
genai.configure(api_key=GEMINI_API_KEY)




GEMINI_MODEL = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config={
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 1024, # Ensure this is sufficient for your reports
    }
)

# --- Helper Functions: AI Interactions ---

@login_required
def get_chat_sessions(request):
    """Returns all chat sessions for sidebar"""
    sessions = ChatSession.objects.filter(user=request.user).values(
        'session_id', 'title', 'created_at', 'updated_at'
    )
    return JsonResponse({'sessions': list(sessions)})

def calculate_health_score(data: dict) -> dict:
    """
    Calculates a comprehensive health score and detailed metrics based on user data.
    Returns a dictionary with overall score, category scores, and analysis.
    """
    scores = {
        'fitness': 0,
        'nutrition': 0,
        'lifestyle': 0,
        'risk_factors': 100
    }
    
    positive_habits = []
    improvement_areas = []
    
    # Calculate BMI
    height_m = float(data['height_cm']) / 100
    weight = float(data['weight_kg'])
    bmi = weight / (height_m ** 2)
    
    # BMI Assessment
    if 18.5 <= bmi < 25:
        scores['fitness'] += 25
        positive_habits.append('Healthy BMI range - great job maintaining your weight!')
    elif 25 <= bmi < 30:
        scores['fitness'] += 15
        improvement_areas.append('BMI slightly elevated - consider gradual weight management')
    elif bmi >= 30:
        scores['fitness'] += 5
        improvement_areas.append('BMI indicates obesity - consult healthcare provider for guidance')
    else:
        scores['fitness'] += 10
        improvement_areas.append('BMI below healthy range - ensure adequate nutrition')
    
    # Exercise Assessment
    exercise_map = {
        'daily': 30,
        '3-4_times_week': 25,
        '1-2_times_week': 15,
        'rarely': 5
    }
    exercise_score = exercise_map.get(data['exercise_frequency'], 5)
    scores['fitness'] += exercise_score
    
    if exercise_score >= 25:
        positive_habits.append(f'Excellent exercise routine - {data["exercise_frequency"].replace("_", " ")}')
    else:
        improvement_areas.append('Increase physical activity to 150 minutes per week')
    
    # Diet Assessment
    diet_map = {
        'excellent': 35,
        'good': 25,
        'average': 15,
        'poor': 5
    }
    diet_score = diet_map.get(data['diet_quality'], 15)
    scores['nutrition'] += diet_score
    
    if diet_score >= 25:
        positive_habits.append('Maintaining a nutritious diet')
    else:
        improvement_areas.append('Focus on whole foods, fruits, and vegetables')
    
    scores['nutrition'] += 20 if diet_score >= 25 else 10
    
    # Lifestyle Assessment - Smoking
    if data['smoking_status'] == 'never':
        scores['lifestyle'] += 40
        positive_habits.append('Non-smoker - excellent for long-term health!')
    elif data['smoking_status'] == 'former':
        scores['lifestyle'] += 30
        positive_habits.append('Successfully quit smoking - keep it up!')
    else:
        scores['lifestyle'] += 5
        improvement_areas.append('Consider smoking cessation programs')
        scores['risk_factors'] -= 30
    
    # Lifestyle Assessment - Alcohol
    alcohol_map = {
        'never': 30,
        'rarely': 25,
        'moderate': 15,
        'heavy': 5
    }
    alcohol_score = alcohol_map.get(data['alcohol_consumption'], 15)
    scores['lifestyle'] += alcohol_score
    
    if alcohol_score >= 25:
        positive_habits.append('Responsible alcohol consumption')
    elif alcohol_score <= 15:
        improvement_areas.append('Reduce alcohol intake to improve health')
        scores['risk_factors'] -= 15
    
    # Risk Factors - Age consideration
    age = int(data['age'])
    if age > 50:
        scores['risk_factors'] -= 10
        improvement_areas.append('Regular health screenings recommended at your age')
    
    # Calculate overall score
    overall_score = int(
        (scores['fitness'] * 0.3) +
        (scores['nutrition'] * 0.25) +
        (scores['lifestyle'] * 0.25) +
        (scores['risk_factors'] * 0.2)
    )
    
    for key in scores:
        scores[key] = min(scores[key], 100)
    
    return {
        'overall_score': overall_score,
        'metrics': scores,
        'positive_habits': positive_habits,
        'improvement_areas': improvement_areas,
        'bmi': round(bmi, 1)
    }


def generate_ai_recommendations(data: dict, score_data: dict) -> list:
    """Uses AI to generate personalized recommendations based on health data."""
    prompt = f"""
Based on this health profile, provide 4-5 specific, actionable recommendations. For each recommendation:
1. Focus on one health aspect (fitness, nutrition, lifestyle, preventive care)
2. Be specific and practical
3. Keep it encouraging and achievable

User Data:
- Age: {data['age']}, Gender: {data['gender']}
- BMI: {score_data['bmi']} (Height: {data['height_cm']}cm, Weight: {data['weight_kg']}kg)
- Exercise: {data['exercise_frequency'].replace('_', ' ')}
- Diet: {data['diet_quality']}
- Smoking: {data['smoking_status']}
- Alcohol: {data['alcohol_consumption']}

Health Score: {score_data['overall_score']}/100

Format each recommendation as:
ICON: [fa-icon-name] | TITLE: [short title] | PRIORITY: [high/medium/low] | DESCRIPTION: [2-3 sentence description]

Example:
ICON: fa-running | TITLE: Boost Your Cardio | PRIORITY: high | DESCRIPTION: Add 2 weekly cardio sessions...
"""
    
    try:
        ai_response = _generate_ai_content(prompt)
        recommendations = []
        
        for line in ai_response.split('\n'):
            if '|' in line and 'ICON:' in line:
                parts = line.split('|')
                if len(parts) == 4:
                    icon = parts[0].replace('ICON:', '').strip()
                    title = parts[1].replace('TITLE:', '').strip()
                    priority = parts[2].replace('PRIORITY:', '').strip().lower()
                    description = parts[3].replace('DESCRIPTION:', '').strip()
                    
                    recommendations.append({
                        'icon': icon,
                        'title': title,
                        'priority': priority,
                        'description': description
                    })
        
        if len(recommendations) < 3:
            recommendations = [
                {
                    'icon': 'fa-running',
                    'title': 'Maintain Regular Exercise',
                    'priority': 'high',
                    'description': 'Continue your current routine and gradually increase intensity. Aim for 150 minutes of moderate activity weekly.'
                },
                {
                    'icon': 'fa-apple-alt',
                    'title': 'Optimize Nutrition',
                    'priority': 'medium',
                    'description': 'Focus on whole foods, lean proteins, and colorful vegetables. Stay hydrated with 8 glasses of water daily.'
                },
                {
                    'icon': 'fa-bed',
                    'title': 'Prioritize Sleep',
                    'priority': 'medium',
                    'description': 'Aim for 7-9 hours of quality sleep. Establish a consistent bedtime routine for better rest.'
                },
                {
                    'icon': 'fa-heartbeat',
                    'title': 'Schedule Regular Checkups',
                    'priority': 'low',
                    'description': 'Book annual health screenings to monitor key health markers and catch issues early.'
                }
            ]
        
        return recommendations[:5]
        
    except Exception as e:
        print(f"Error generating AI recommendations: {e}")
        return []


@login_required
def health_dashboard_view(request):
    dashboard_data = request.session.get('health_dashboard_data')
    
    if not dashboard_data:
        return redirect('chat_room')
    
    scores_data = dashboard_data['scores']
    
    context = {
        'user_data': dashboard_data['user_data'],
        'scores': scores_data,
        'recommendations': dashboard_data['recommendations'],
        'generated_at': dashboard_data['generated_at'],
        'overall_score': scores_data['overall_score'],
        'fitness_score': scores_data['metrics']['fitness'],
        'nutrition_score': scores_data['metrics']['nutrition'],
        'lifestyle_score': scores_data['metrics']['lifestyle'],
        'risk_factors_score': scores_data['metrics']['risk_factors'],
        'bmi_value': scores_data['bmi'],
    }
        
    return render(request, 'health_dashboard.html', context)

def _generate_ai_content(prompt: str) -> str:
    """Handles interaction with the Gemini model and returns stripped text."""
    try:
        response = GEMINI_MODEL.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        # Log the error for debugging, but return a user-friendly message
        print(f"Gemini API error: {e}")
        return "Sorry, I'm having trouble connecting right now. Please try again later."

def handle_general_health(message: str) -> str:
    """Provides general health advice using AI."""
    prompt = (
        "You are Curaid, a friendly general health assistant. Provide concise, accurate health advice based on the user’s query. "
        "For specific needs, suggest using the mode selector. Keep responses natural and supportive.\n\n"
        f"Query: {message}"
    )
    return _generate_ai_content(prompt)

def handle_symptoms_predictor(input_data: str) -> str:
    """Analyzes symptoms using AI."""
    prompt = (
        "You are Curaid, a health assistant. Analyze the user's input and determine if the described symptoms suggest a health concern. "
        "If age is provided, consider it in your assessment. Provide a concise response with a recommendation (e.g., 'Healthy: Symptoms seem mild. Stay hydrated and monitor.' or 'Possible risk: Symptoms suggest a concern. Consult a doctor.'). "
        "If the input is unstructured or lacks age, suggest the user provide age and symptoms in the format 'age: 30, symptoms: fatigue, headache'. "
        f"Input: {input_data}"
    )
    return _generate_ai_content(prompt)

def handle_mental_health(input_data: str, user: CustomUser) -> str:
    """Provides mental health support and suggests activities using AI."""
    prompt = (
        "You are Curaid, a mental health assistant. Provide supportive advice based on the user’s input (e.g., feelings like 'sadness' or 'stress'). "
        "If appropriate, **suggest a specific actionable mental well-being activity** that the user could add to their daily routine, like 'meditation', 'mindful breathing', 'short walk', 'journaling', or 'reaching out to a friend'. "
        "Format the suggestion clearly, for example: 'I suggest you try [activity] today.' Keep it friendly and concise.\n\n"
        f"Query: {input_data}"
    )
    curo_response = _generate_ai_content(prompt)

    # Check for suggested activity from AI response
    suggested_activity = None
    if "i suggest you try" in curo_response.lower():
        if "meditation" in curo_response.lower():
            suggested_activity = "meditation"
        elif "mindful breathing" in curo_response.lower() or "breathing exercise" in curo_response.lower():
            suggested_activity = "mindful breathing"
        elif "walk" in curo_response.lower():
            suggested_activity = "short walk"
        elif "journaling" in curo_response.lower():
            suggested_activity = "journaling"
        elif "reaching out to a friend" in curo_response.lower():
            suggested_activity = "reach out to a friend"
        # Add more keywords for activities here as needed

    if suggested_activity:
        # Store in session for follow-up
        user.session['last_suggested_activity'] = suggested_activity
        curo_response += f"\n\nWould you like me to add '{suggested_activity}' to your to-do list? (Say 'yes' or 'no')"
    return curo_response

def handle_disease_predictor(input_data: str, disease_type: str) -> str:
    """Predicts disease risk using loaded ML models."""
    try:
        inputs = {}
        for part in input_data.split(","):
            if ":" not in part:
                continue
            key, value = part.split(":")
            key = key.strip()
            value = value.strip()
            if key == "disease": # Skip if 'disease' key is present
                continue
            inputs[key] = float(value)

        if disease_type == "diabetes":
            required_keys = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree_function', 'age']
            if not all(k in inputs for k in required_keys):
                raise ValueError("Missing required inputs for diabetes prediction.")
            model_input = [inputs[k] for k in required_keys]
            prediction = DIABETES_MODEL.predict(np.array(model_input).reshape(1, -1))
            if prediction[0] == 0:
                return "You are likely healthy regarding diabetes. Maintain a balanced diet and exercise 30 minutes daily."
            return "There is a risk of diabetes. Consult a doctor and reduce refined carbs."
        
        elif disease_type == "heart":
            required_keys = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
            if not all(k in inputs for k in required_keys):
                raise ValueError("Missing required inputs for heart disease prediction.")
            model_input = [inputs[k] for k in required_keys]
            prediction = HEART_MODEL.predict(np.array(model_input).reshape(1, -1))
            if prediction[0] == 0:
                return "You are likely healthy regarding heart disease. Try regular cardio and a veggie-rich diet."
            return "There is a risk of heart disease. See a doctor and consider lowering salt intake."
        
        else:
            return "Invalid disease type selected for prediction."
    except ValueError as ve:
        return f"Error: {ve}. Please provide inputs in the correct format. For diabetes: 'pregnancies: 2, glucose: 120, ..., age: 30'. For heart: 'age: 50, sex: 1, ..., thal: 3'."
    except Exception as e:
        return f"An unexpected error occurred during prediction: {e}"

# --- Helper Functions: To-Do Management ---

def _format_todos_for_response(user: CustomUser) -> str:
    """Formats a user's to-do list for display in chat."""
    todos = UserTodo.objects.filter(user=user).order_by('completed', 'due_date', '-created_at')
    if not todos:
        return "Your to-do list is empty! Want to add something? (e.g., 'Add todo: Meditate')"

    todo_list_text = "Here's your to-do list:\n"
    for i, todo in enumerate(todos):
        status = "[Done]" if todo.completed else "[Pending]"
        due_info = ""
        if todo.due_date:
            if todo.due_date == date.today():
                due_info = " (Due Today)"
            elif todo.due_date == date.today() + timedelta(days=1):
                due_info = " (Due Tomorrow)"
            else:
                due_info = f" (Due: {todo.due_date.strftime('%b %d')})"
        todo_list_text += f"{i+1}. {status} {todo.task_description}{due_info}\n"
    return todo_list_text

def _serialize_todos_for_json(user: CustomUser) -> list:
    """Serializes user todos for JSON response."""
    todos_queryset = UserTodo.objects.filter(user=user).order_by('completed', 'due_date', '-created_at')
    return [
        {
            'id': todo.id,
            'task_description': todo.task_description,
            'completed': todo.completed,
            'due_date': todo.due_date.isoformat() if todo.due_date else None
        } for todo in todos_queryset
    ]

def handle_todo_commands(request, message: str) -> str:
    """Processes user commands related to the to-do list."""
    user = request.user
    message_lower = message.lower()
    response_message = ""

    # Handle "yes/no" after an AI suggestion
    last_suggested_activity = request.session.get('last_suggested_activity')
    if message_lower == "yes" and last_suggested_activity:
        UserTodo.objects.create(
            user=user,
            task_description=last_suggested_activity,
            suggested_by_curo=True,
            due_date=date.today()
        )
        response_message = f"Added '{last_suggested_activity}' to your list for today! "
        del request.session['last_suggested_activity']
        response_message += "\n" + _format_todos_for_response(user)
        return response_message
    elif message_lower == "no" and last_suggested_activity:
        response_message = "Okay, no problem! Let me know if you change your mind."
        del request.session['last_suggested_activity']
        return response_message

    # Handle explicit To-Do Commands
    if message_lower.startswith("add todo:"):
        task_description = message_lower.replace("add todo:", "").strip()
        if task_description:
            due_date = date.today() + timedelta(days=1) # Default to tomorrow
            UserTodo.objects.create(
                user=user,
                task_description=task_description,
                suggested_by_curo=False,
                due_date=due_date
            )
            response_message = f"Okay, I've added '{task_description}' to your list!"
            response_message += "\n" + _format_todos_for_response(user)
        else:
            response_message = "Please tell me what to add. Example: 'Add todo: Meditate for 10 mins'."

    elif message_lower.startswith(("done todo:", "complete todo:")):
        identifier = message_lower.split(":", 1)[1].strip() if ":" in message_lower else ""
        todo_item = None
        try:
            todos = UserTodo.objects.filter(user=user, completed=False).order_by('due_date', '-created_at')
            todo_index = int(identifier) - 1
            if 0 <= todo_index < len(todos):
                todo_item = todos[todo_index]
        except (ValueError, IndexError):
            todo_item = UserTodo.objects.filter(
                user=user,
                task_description__icontains=identifier,
                completed=False
            ).first()

        if todo_item:
            todo_item.completed = True
            todo_item.completed_at = datetime.now()
            todo_item.save()
            response_message = f"Great job! You've completed: '{todo_item.task_description}'."
            response_message += "\n" + _format_todos_for_response(user)
        else:
            response_message = f"Couldn't find an incomplete task matching '{identifier}'. Try 'list todos' to see numbers."
    
    elif message_lower == "list todos" or message_lower == "my todos":
        response_message = _format_todos_for_response(user)
    
    elif message_lower == "clear todos":
        UserTodo.objects.filter(user=user, completed=True).delete()
        response_message = "Completed tasks have been cleared."
        response_message += "\n" + _format_todos_for_response(user)

    return response_message or "I didn't understand that to-do command. Try 'Add todo: [task]', 'Done todo: [task or number]', 'List todos', or 'Clear todos'."


# --- Django Views ---

@login_required
@login_required
def chat_room(request):
    """Handles the main chat room logic, including AI interaction and to-do management."""
    
    # Get or create current session
    current_session_id = request.session.get('current_chat_session_id')
    if current_session_id:
        try:
            current_session = ChatSession.objects.get(session_id=current_session_id, user=request.user)
        except ChatSession.DoesNotExist:
            current_session = None
    else:
        current_session = None

    # Load messages only from current session
    if current_session:
        chat_history_list = list(ChatMessage.objects.filter(
            user=request.user,
            session=current_session
        ).order_by('timestamp').values('message', 'bot_response', 'timestamp'))
    else:
        chat_history_list = []

    # Get/Set AI mode and disease type from session or POST
    ai_mode = request.POST.get('ai-mode', request.session.get('ai_mode', 'general'))
    disease_type = request.POST.get('disease-type', request.session.get('disease_type', 'diabetes'))
    request.session['ai_mode'] = ai_mode
    request.session['disease_type'] = disease_type

    if request.method == 'POST':
        message = request.POST.get('message', '').strip()
        uploaded_file = request.FILES.get('health_report')
        new_ai_mode = request.POST.get('ai-mode', '').strip()
        new_disease_type = request.POST.get('disease-type', '').strip()

        # Update mode if changed by user in form
        if new_ai_mode:
            ai_mode = new_ai_mode
            request.session['ai_mode'] = ai_mode
        if new_disease_type:
            disease_type = new_disease_type
            request.session['disease_type'] = disease_type

        bot_response = ""

        # Create session if none exists
        if not current_session:
            current_session = ChatSession.objects.create(user=request.user)
            request.session['current_chat_session_id'] = str(current_session.session_id)
            # Generate title from first message
            if message and len(message) > 5:
                current_session.title = message[:50] + "..." if len(message) > 50 else message
                current_session.save()

        # Handle file upload
        if uploaded_file:
            bot_response = handle_health_report_upload(uploaded_file, request.user)
        # Handle to-do commands
        elif message.lower().startswith(("add todo:", "done todo:", "complete todo:", "list todos", "my todos", "clear todos")) or \
             (message.lower() in ["yes", "no"] and request.session.get('last_suggested_activity')):
            bot_response = handle_todo_commands(request, message)
        else:
            # Handle general chat/AI modes
            greetings = ["hi", "hello", "hey", "greetings", "yo", "what's up", "howdy"]
            if message.lower() in greetings:
                bot_response = "Hey there! I'm Curo, your friendly health coach. Ask me about health or use the mode selector for specific help!"
            elif not message or len(message.split()) <= 2:
                bot_response = "Hey! I'm Curo, ready to help with your health questions. Try asking something specific or select a mode!"
            else:
                if ai_mode == "general":
                    bot_response = handle_general_health(message)
                elif ai_mode == "symptoms":
                    bot_response = handle_symptoms_predictor(message)
                elif ai_mode == "mental":
                    bot_response = handle_mental_health(message, request.user)
                elif ai_mode == "disease":
                    bot_response = handle_disease_predictor(message, disease_type)
                else:
                    bot_response = "Invalid mode selected. Please choose a valid mode."

        # Save current interaction
        ChatMessage.objects.create(
            user=request.user,
            session=current_session,
            message=message,
            bot_response=bot_response,
            timestamp=datetime.now()
        )
        current_session.updated_at = datetime.now()
        current_session.save()

        # For AJAX requests, return JSON with updated todos
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            todos_data = _serialize_todos_for_json(request.user)
            return JsonResponse({'bot_response': bot_response, 'todos': todos_data})

        # For full page reloads, re-render the template
        active_todos_json_for_template = json.dumps(_serialize_todos_for_json(request.user))
        chat_sessions = ChatSession.objects.filter(user=request.user)
        return render(request, 'chat.html', {
            'chat_history': chat_history_list,
            'chat_sessions': chat_sessions,
            'ai_mode': ai_mode,
            'disease_type': disease_type,
            'active_todos': active_todos_json_for_template
        })

    # Initial GET request to load the chat page
    active_todos_json_for_template = json.dumps(_serialize_todos_for_json(request.user))
    chat_sessions = ChatSession.objects.filter(user=request.user)
    return render(request, 'chat.html', {
        'chat_history': chat_history_list,
        'chat_sessions': chat_sessions,
        'ai_mode': ai_mode,
        'disease_type': disease_type,
        'active_todos': active_todos_json_for_template
    })




def handle_health_report_upload(uploaded_file, user: CustomUser) -> str:
    """Analyzes uploaded health report using AI."""
    try:
        # Extract text based on file type
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext == 'pdf':
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            extracted_text = ""
            for page in pdf_reader.pages:
                extracted_text += page.extract_text()
        
        elif file_ext in ['jpg', 'jpeg', 'png']:
            image = Image.open(uploaded_file)
            extracted_text = pytesseract.image_to_string(image)
        
        elif file_ext == 'txt':
            extracted_text = uploaded_file.read().decode('utf-8')
        
        else:
            return "Unsupported file type. Please upload PDF, JPG, PNG, or TXT."
        
        # Analyze with AI
        prompt = f"""
        You are Curaid, a health report analyzer. Analyze this health report and provide:
        1. Key findings (2-3 bullet points)
        2. Health metrics assessment
        3. Recommendations (2-3 actionable items)
        
        Keep it concise and supportive. If values seem concerning, suggest consulting a doctor.
        
        Report Content:
        {extracted_text[:3000]}  # Limit to avoid token limits
        """
        
        return _generate_ai_content(prompt)
        
    except Exception as e:
        print(f"Error processing health report: {e}")
        return "I had trouble reading that report. Please ensure it's a clear, readable file."
    

@csrf_exempt # Consider making this POST-only with csrf_token in AJAX for better security
@login_required
def generate_health_report(request):
    """Generates a comprehensive health dashboard with scores and recommendations."""
    if request.method == 'POST':
        try:
            data = {
                'age': request.POST.get('age'),
                'gender': request.POST.get('gender'),
                'height_cm': request.POST.get('height_cm'),
                'weight_kg': request.POST.get('weight_kg'),
                'exercise_frequency': request.POST.get('exercise_frequency'),
                'diet_quality': request.POST.get('diet_quality'),
                'smoking_status': request.POST.get('smoking_status'),
                'alcohol_consumption': request.POST.get('alcohol_consumption')
            }
            
            score_data = calculate_health_score(data)
            recommendations = generate_ai_recommendations(data, score_data)
            
            print("DEBUG - Score Data:", score_data)  # Add this line
            print("DEBUG - Recommendations:", recommendations) 

            request.session['health_dashboard_data'] = {
                'user_data': data,
                'scores': score_data,
                'recommendations': recommendations,
                'generated_at': datetime.now().isoformat()
            }
            request.session.modified = True  # Force session save
            
            return JsonResponse({
                'status': 'success',
                'redirect_url': '/chat/health-dashboard/'
            })

        except Exception as e:
            print(f"Error generating health report: {e}")
            return JsonResponse({
                'status': 'error',
                'message': f'Failed to generate report: {str(e)}'
            }, status=500)
    else:
        return JsonResponse({
            'status': 'error',
            'message': 'Invalid request method.'
        }, status=405)


@login_required
def test_health_calc(request):
    """Temporary test endpoint"""
    test_data = {
        'age': '30',
        'gender': 'male',
        'height_cm': '175',
        'weight_kg': '70',
        'exercise_frequency': 'daily',
        'diet_quality': 'good',
        'smoking_status': 'never',
        'alcohol_consumption': 'rarely'
    }
    
    score_data = calculate_health_score(test_data)
    print("Test Score Data:", score_data)
    
    return JsonResponse(score_data)
    
@login_required
def profile_view(request):
    """Handles user profile updates (email and password)."""
    if request.method == 'POST':
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            email = request.POST.get('email')
            password = request.POST.get('password')

            try:
                user = request.user
                if email and email != user.email:
                    if CustomUser.objects.filter(email=email).exclude(id=user.id).exists():
                        return JsonResponse({'success': False, 'error': 'Email already in use.'})
                    user.email = email

                if password:
                    user.set_password(password)
                user.save()
                return JsonResponse({'success': True})
            except Exception as e:
                return JsonResponse({'success': False, 'error': str(e)})
        return JsonResponse({'success': False, 'error': 'Invalid request'})
    return render(request, 'profile.html', {'user': request.user})

def signup_view(request):
    """Handles user registration."""
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')

        if CustomUser.objects.filter(email=email).exists():
            return render(request, 'login.html', {
                'signup_errors': 'Email already exists. Try logging in or use a different email.'
            })

        user = CustomUser.objects.create_user(
            email=email,
            username=name, # Using name for username, adjust if your CustomUser handles this differently
            password=password,
            first_name=name # Storing name as first_name
        )
        user.save()
        login(request, user)
        return redirect('landing')
    return render(request, 'login.html')


@login_required
def delete_chat_session(request, session_id):
    """Deletes a specific chat session"""
    if request.method == 'POST':
        try:
            session = ChatSession.objects.get(session_id=session_id, user=request.user)
            
            # If deleting current session, clear it from session
            if str(session.session_id) == request.session.get('current_chat_session_id'):
                del request.session['current_chat_session_id']
            
            session.delete()
            return JsonResponse({'status': 'success'})
        except ChatSession.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Session not found'}, status=404)
    return JsonResponse({'status': 'error', 'message': 'Invalid method'}, status=405)


@login_required
def logout_view(request):
    """Logs out the current user."""
    if request.method == 'POST':
        logout(request)
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'success': True})
        return redirect('login')
    return redirect('login') # Handles direct GET requests to logout (redirects to login)

def index_view(request):
    """Renders the main index page."""
    return render(request, 'index.html')

def landing_view(request):
    """Renders the landing page, redirects to login if not authenticated."""
    if not request.user.is_authenticated:
        return redirect('login')
    return render(request, 'landing.html')

def mark_todo_done(request, todo_id):
    # Your existing logic to mark todo as done
    if request.method == 'POST':
        try:
            # Get the Todo item
            todo = UserTodo.objects.get(id=todo_id)
            # Parse the JSON body
            import json
            data = json.loads(request.body)
            completed_status = data.get('completed')

            # Update the todo status
            todo.completed = completed_status
            todo.save()

            # Return all todos
            todos = list(UserTodo.objects.values()) # Get all todos, including the updated one
            return JsonResponse({'todos': todos})
        except UserTodo.DoesNotExist:
            return JsonResponse({'error': 'Todo not found'}, status=404)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=405)


def clear_completed_todos(request):
    # Your existing logic to clear completed todos
    if request.method == 'POST':
        try:
            UserTodo.objects.filter(completed=True).delete()
            todos = list(UserTodo.objects.values()) # Get remaining todos
            return JsonResponse({'todos': todos})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=405)


@login_required
def new_chat_session(request):
    """Creates a new chat session"""
    if 'current_chat_session_id' in request.session:
        del request.session['current_chat_session_id']
    return JsonResponse({'status': 'success'})

@login_required
def load_chat_session(request, session_id):
    """Loads a specific chat session"""
    try:
        session = ChatSession.objects.get(session_id=session_id, user=request.user)
        request.session['current_chat_session_id'] = str(session.session_id)
        
        messages = ChatMessage.objects.filter(session=session).order_by('timestamp').values(
            'message', 'bot_response', 'timestamp'
        )
        
        return JsonResponse({
            'status': 'success',
            'messages': list(messages),
            'session_title': session.title
        })
    except ChatSession.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Session not found'}, status=404)


