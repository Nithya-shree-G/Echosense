from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import os
from datetime import datetime
from dotenv import load_dotenv
import json
from transformers import pipeline

load_dotenv()

app = Flask(__name__)
CORS(app)

# Load Hugging Face emotion model (runs locally, free forever)
print("Loading emotion analysis model (first time may take 2-3 minutes)...")
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)
print("Emotion model loaded successfully!")

# Database Configuration
DATABASE = 'echosense.db'

def get_db():
    """Create database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database with required tables"""
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active INTEGER DEFAULT 1
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            sender TEXT NOT NULL,
            message TEXT NOT NULL,
            emotion_data TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS emotion_analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            sadness REAL DEFAULT 0,
            anger REAL DEFAULT 0,
            anxiety REAL DEFAULT 0,
            loneliness REAL DEFAULT 0,
            gratitude REAL DEFAULT 0,
            joy REAL DEFAULT 0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    ''')
    
    conn.commit()
    conn.close()

init_db()

def analyze_emotion_with_ai(text):
    """
    FREE emotion analysis using Hugging Face Transformers
    No API key needed, runs completely locally
    """
    try:
        print(f"\n=== EMOTION ANALYSIS (Hugging Face) ===")
        print(f"Analyzing: {text[:100]}...")
        
        # Get predictions from the model
        results = emotion_classifier(text)[0]
        
        # The model returns: anger, disgust, fear, joy, neutral, sadness, surprise
        emotion_scores = {}
        for result in results:
            emotion_scores[result['label']] = result['score']
        
        print(f"Raw scores: {emotion_scores}")
        
        # Map model emotions to our 6 categories
        mapped_emotions = {
            'sadness': emotion_scores.get('sadness', 0),
            'anger': emotion_scores.get('anger', 0),
            'anxiety': emotion_scores.get('fear', 0),  # fear maps to anxiety
            'loneliness': emotion_scores.get('sadness', 0) * 0.6,  # derive from sadness
            'gratitude': max(emotion_scores.get('joy', 0) * 0.3, 0.1) if emotion_scores.get('joy', 0) > 0.5 else 0,
            'joy': emotion_scores.get('joy', 0)
        }
        
        # Find primary emotion
        primary_emotion = max(mapped_emotions, key=mapped_emotions.get)
        confidence = mapped_emotions[primary_emotion]
        
        # If confidence is too low, mark as neutral
        if confidence < 0.25:
            primary_emotion = 'neutral'
            confidence = 0.3
        
        result = {
            'sadness': round(mapped_emotions['sadness'], 3),
            'anger': round(mapped_emotions['anger'], 3),
            'anxiety': round(mapped_emotions['anxiety'], 3),
            'loneliness': round(mapped_emotions['loneliness'], 3),
            'gratitude': round(mapped_emotions['gratitude'], 3),
            'joy': round(mapped_emotions['joy'], 3),
            'primary_emotion': primary_emotion,
            'confidence': round(confidence, 3),
            'reasoning': f'AI detected {primary_emotion} with {confidence*100:.0f}% confidence'
        }
        
        print(f"Primary emotion: {primary_emotion} ({confidence*100:.0f}%)")
        print("=== ANALYSIS COMPLETE ===\n")
        
        return result
        
    except Exception as e:
        print(f"Emotion analysis error: {e}")
        return {
            'sadness': 0.0,
            'anger': 0.0,
            'anxiety': 0.0,
            'loneliness': 0.0,
            'gratitude': 0.0,
            'joy': 0.0,
            'primary_emotion': 'neutral',
            'confidence': 0.1,
            'reasoning': 'Analysis unavailable'
        }

def generate_empathetic_response(user_message, emotion_data):
    """
    Generate contextual empathetic responses based on detected emotion
    """
    primary_emotion = emotion_data.get('primary_emotion', 'neutral')
    confidence = emotion_data.get('confidence', 0)
    
    # Base responses for each emotion
    responses = {
        'sadness': [
            "I can hear the sadness in your words, and I want you to know that these feelings are completely valid. You're not alone in experiencing this.",
            "It sounds like you're going through a difficult time. Your feelings of sadness deserve acknowledgment and care.",
            "The pain you're experiencing is real. Thank you for trusting me with these feelings."
        ],
        'anger': [
            "I sense your frustration and anger. Those feelings are real and deserve acknowledgment. What's at the heart of this for you?",
            "Your anger is valid. It often signals that something important needs attention or change.",
            "I hear the intensity of what you're feeling. Anger can be a powerful signal that your boundaries or values are being challenged."
        ],
        'anxiety': [
            "The anxiety you're feeling sounds really overwhelming. Remember to take deep breaths - you're going to get through this moment.",
            "I can sense the worry in your words. Anxiety can feel all-consuming, but you're taking a positive step by expressing it.",
            "What you're experiencing sounds stressful. It's okay to feel anxious, and it's brave of you to acknowledge it."
        ],
        'loneliness': [
            "Feeling isolated is incredibly difficult. Thank you for sharing this with me. Connection matters, and I'm here to listen.",
            "Loneliness can be so painful. Please know that reaching out, as you're doing now, is an act of courage.",
            "The sense of being alone that you're describing is real and valid. You deserve connection and understanding."
        ],
        'gratitude': [
            "Your gratitude and appreciation really shine through your words. It's beautiful to see you recognizing the positive.",
            "The thankfulness you're expressing is wonderful. Gratitude has a way of opening our hearts to more goodness.",
            "I love hearing the appreciation in your message. It's clear you're noticing the good around you."
        ],
        'joy': [
            "I can feel the happiness in your message! Your joy is wonderful and worth celebrating. Tell me more about what's bringing you this happiness!",
            "The excitement and joy in your words is contagious! It's beautiful to witness your happiness.",
            "Your joy really comes through! These moments of happiness are precious - savor this feeling."
        ],
        'neutral': [
            "Thank you for sharing your thoughts with me. I'm here to listen and support you. What else is on your mind?",
            "I'm here with you. Feel free to share whatever you're thinking or feeling.",
            "I appreciate you opening up. What would be most helpful for you to explore right now?"
        ]
    }
    
    # Select a response based on primary emotion
    import random
    response_list = responses.get(primary_emotion, responses['neutral'])
    return random.choice(response_list)

def generate_supportive_action(user_message, emotion_data):
    """
    Generate supportive suggestions based on emotion
    """
    primary_emotion = emotion_data.get('primary_emotion', 'neutral')
    confidence = emotion_data.get('confidence', 0)
    
    if confidence < 0.3:
        return ""
    
    actions = {
        'sadness': [
            "Try writing down three small things you're grateful for, even if they seem insignificant right now.",
            "Consider reaching out to someone you trust - connection can help ease sadness.",
            "Give yourself permission to feel this sadness without judgment. It's part of being human."
        ],
        'anger': [
            "Take five slow, deep breaths - inhale for 4 counts, hold for 4, exhale for 6.",
            "Try physical movement - a walk, stretching, or exercise can help process anger.",
            "Write down what's making you angry without filtering. Sometimes getting it out helps."
        ],
        'anxiety': [
            "Use the 5-4-3-2-1 grounding technique: name 5 things you see, 4 you hear, 3 you touch, 2 you smell, 1 you taste.",
            "Focus on slow, deep breathing. Breathe in for 4 counts, hold for 4, out for 6. Repeat 5 times.",
            "Ask yourself: 'What can I control right now?' Then focus only on that."
        ],
        'loneliness': [
            "Send a thoughtful message to someone you care about - even a simple 'thinking of you' can rebuild connection.",
            "Join an online community or group around something you're interested in.",
            "Do something kind for yourself, treating yourself with the compassion you'd show a friend."
        ],
        'gratitude': [
            "Take a moment to write about why you're grateful for this particular thing or person in detail.",
            "Share your gratitude with someone directly - let them know the positive impact they've had.",
            "Reflect on how this gratitude makes you feel and try to carry that feeling forward."
        ],
        'joy': [
            "Take a moment to fully savor and appreciate this happy feeling without rushing past it.",
            "Share your joy with someone close to you - happiness multiplies when shared.",
            "Write down what's bringing you joy so you can revisit this memory later."
        ]
    }
    
    import random
    action_list = actions.get(primary_emotion, [])
    return random.choice(action_list) if action_list else ""

@app.route('/api/session/create', methods=['POST'])
def create_session():
    """Create new session"""
    data = request.json
    user_id = data.get('user_id')
    session_id = data.get('session_id')
    
    if not user_id or not session_id:
        return jsonify({'error': 'user_id and session_id required'}), 400
    
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            'INSERT INTO sessions (session_id, user_id) VALUES (?, ?)',
            (session_id, user_id)
        )
        conn.commit()
        return jsonify({'message': 'Session created', 'session_id': session_id}), 201
    except sqlite3.IntegrityError:
        return jsonify({'message': 'Session exists', 'session_id': session_id}), 200
    finally:
        conn.close()

@app.route('/api/session/close', methods=['POST'])
def close_session():
    """Close session and delete data"""
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({'error': 'session_id required'}), 400
    
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM emotion_analytics WHERE session_id = ?', (session_id,))
        cursor.execute('DELETE FROM sessions WHERE session_id = ?', (session_id,))
        
        conn.commit()
        return jsonify({'message': 'Session closed and data deleted'}), 200
    finally:
        conn.close()

@app.route('/api/message/send', methods=['POST'])
def send_message():
    """Process message with FREE AI emotion analysis"""
    data = request.json
    session_id = data.get('session_id')
    user_message = data.get('message')
    
    if not session_id or not user_message:
        return jsonify({'error': 'session_id and message required'}), 400
    
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        # FREE AI emotion analysis using Hugging Face
        emotion_data = analyze_emotion_with_ai(user_message)
        
        cursor.execute(
            'INSERT INTO messages (session_id, sender, message, emotion_data) VALUES (?, ?, ?, ?)',
            (session_id, 'user', user_message, json.dumps(emotion_data))
        )
        
        cursor.execute(
            '''INSERT INTO emotion_analytics 
            (session_id, sadness, anger, anxiety, loneliness, gratitude, joy) 
            VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (session_id, 
             emotion_data.get('sadness', 0),
             emotion_data.get('anger', 0),
             emotion_data.get('anxiety', 0),
             emotion_data.get('loneliness', 0),
             emotion_data.get('gratitude', 0),
             emotion_data.get('joy', 0))
        )
        
        # Generate response and suggestion
        ai_response = generate_empathetic_response(user_message, emotion_data)
        supportive_action = generate_supportive_action(user_message, emotion_data)
        
        cursor.execute(
            'INSERT INTO messages (session_id, sender, message, emotion_data) VALUES (?, ?, ?, ?)',
            (session_id, 'bot', ai_response, json.dumps(emotion_data))
        )
        
        conn.commit()
        
        return jsonify({
            'response': ai_response,
            'emotion_data': emotion_data,
            'supportive_action': supportive_action
        }), 200
        
    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
        return jsonify({'error': 'Failed to process message'}), 500
    finally:
        conn.close()

@app.route('/api/session/stats', methods=['GET'])
def get_session_stats():
    """Get stats"""
    session_id = request.args.get('session_id')
    
    if not session_id:
        return jsonify({'error': 'session_id required'}), 400
    
    conn = get_db()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            'SELECT COUNT(*) as count FROM messages WHERE session_id = ? AND sender = "user"',
            (session_id,)
        )
        message_count = cursor.fetchone()['count']
        
        cursor.execute(
            '''SELECT sadness, anger, anxiety, loneliness, gratitude, joy 
            FROM emotion_analytics WHERE session_id = ? 
            ORDER BY timestamp DESC LIMIT 1''',
            (session_id,)
        )
        emotions = cursor.fetchone()
        
        emotion_dict = dict(emotions) if emotions else {
            'sadness': 0, 'anger': 0, 'anxiety': 0,
            'loneliness': 0, 'gratitude': 0, 'joy': 0
        }
        
        return jsonify({
            'message_count': message_count,
            'emotions': emotion_dict
        }), 200
        
    finally:
        conn.close()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy', 
        'service': 'EchoSense AI API (Hugging Face Free)',
        'ai_model': 'emotion-english-distilroberta-base'
    }), 200

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🧠 EchoSense Backend - FREE AI Edition")
    print("="*50)
    print("✅ Using Hugging Face Transformers (100% Free)")
    print("✅ No API keys needed")
    print("✅ Runs completely offline after model download")
    print("="*50)
    print(f"Backend: http://localhost:5000")
    print(f"Health: http://localhost:5000/api/health")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)