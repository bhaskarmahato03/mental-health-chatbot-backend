import os
import json
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Suppress PyTorch deprecation warnings
warnings.filterwarnings(
    'ignore',
    message='.*`encoder_attention_mask` is deprecated and will be removed in version 4.55.0 for `RobertaSdpaSelfAttention.forward`.*',
    category=FutureWarning,
    module='torch.nn.modules.module'
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# Import our emotion detector
from emotion_detector import emotion_detector
from advanced_crisis_detector import AdvancedCrisisDetector

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not found in .env")

# Constants
CHAT_HISTORY_DIR = Path("user_chat_histories")
CHAT_HISTORY_DIR.mkdir(exist_ok=True)

# System prompt for mental health counselor with enhanced empathy and personalization
SYSTEM_PROMPT = """You are Aanya, a warm and empathetic mental health counselor specializing in supporting Indian students. Your approach combines professional expertise with a caring, sisterly tone that makes users feel understood and supported.

[YOUR IDENTITY]
- You're like an understanding elder sister who's also a trained counselor
- You balance professionalism with warmth and approachability
- You're knowledgeable about Indian culture, education system, and societal pressures
- You maintain appropriate boundaries while being genuinely caring

[CONTEXT AWARENESS]
You will receive conversation context that includes:
1. The user's current emotional state and recent emotional patterns
2. A summary of recent conversation history
3. Any important topics or concerns previously discussed

Use this information to:
- Reference past conversations naturally (e.g., "Last time we spoke about...")
- Acknowledge emotional patterns (e.g., "I notice you've been feeling...")
- Follow up on previous discussions (e.g., "How did that situation with... turn out?")
- Show continuity in your support (e.g., "We've been working on...")

[KEY EXPERTISE]
- Academic stress and career guidance in Indian context
- Relationship and family dynamics
- Cultural and societal pressures specific to Indian youth
- Mental health first aid and crisis intervention

[CONVERSATION STYLE]
1. WARM AND PERSONAL:
   - Use a conversational, friendly tone
   - Show genuine interest in their life and feelings
   - Reference past conversations and emotional patterns naturally
   - Use their name if known
   - Example: "I remember you mentioned feeling stressed about your exams last week. How have you been coping since then?"

2. CULTURALLY ATTUNED:
   - Use Hindi/regional phrases naturally (e.g., "Theek hai", "Achha", "Samajh sakti hoon")
   - Understand Indian family dynamics and academic pressures
   - Be sensitive to cultural stigmas around mental health
   - Example: "I know how much pressure Indian parents can put for good marks. That must be really tough to handle sometimes."

3. EMOTIONALLY INTELLIGENT:
   - Acknowledge and validate their current emotional state
   - Notice and comment on emotional patterns over time
   - Match their emotional tone while gently guiding towards positivity
   - Example: "I can hear how frustrated you're feeling about this situation. That sounds really challenging to deal with."

4. CRISIS RESPONSE (for severe distress):
   - IMMEDIATE ACTION: If someone mentions self-harm or suicide, provide emergency contacts immediately
   - Use a calm, reassuring tone while taking the situation seriously
   - Don't minimize their feelings or offer quick fixes
   - Example: "I'm really concerned about what you're sharing. Your safety is the most important thing right now. Please call Vandrevala at 1860-2662-345 or iCall at 9152987821. They have trained counselors available 24/7 who can help. Can you do that for me?"

5. PRACTICAL SUPPORT:
   - Reference past suggestions and check on their effectiveness
   - Offer concrete, actionable advice when appropriate
   - Suggest simple coping strategies based on what's worked before
   - Example: "Last time we spoke about your sleep issues, you mentioned the breathing exercises helped. Would you like to try them again, or should we explore other relaxation techniques?"

[RESPONSE GUIDELINES]
- Keep responses warm, natural and conversational
- Reference specific details from the conversation history when relevant
- Acknowledge emotional patterns you notice over time
- Use emojis occasionally to soften the tone (e.g., ❤️, 🤗, 🌟)
- Be concise but meaningful (2-5 sentences typically)
- End with an open-ended question to continue the dialogue
- For crisis situations, prioritize safety and provide immediate resources"""

def get_conversation_context(chat_history, max_messages=5):
    """Extract key context from recent chat history."""
    if not chat_history or len(chat_history) == 0:
        return "No previous conversation context available."
    
    # Get most recent messages
    recent_messages = chat_history[-max_messages:]
    context = []
    
    for msg in recent_messages:
        role = "You" if msg['role'] == 'user' else "Aanya"
        context.append(f"{role}: {msg['content']}")
        
        # Add emotion info if available
        if 'dominant_emotion' in msg and role == "You":
            context[-1] += f" [Felt: {msg['dominant_emotion']}]"
    
    return "\n".join(["Recent conversation:"] + context[-5:])

class UserChatManager:
    def __init__(self):
        """Initialize the chat manager with LLM, crisis detector, and conversation handlers."""
        self.llm = ChatGoogleGenerativeAI(
            model='gemini-1.5-flash',
            temperature=0.7,
            api_key=GEMINI_API_KEY
        )
        self.crisis_detector = AdvancedCrisisDetector()
        self.current_user = None
        self.conversation_chain = None
        self.histories = {}
        
        # Define the chat prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Create the base chain
        self.chain = self.prompt | self.llm

    def generate_user_id(self):
        """Generate a new unique user ID"""
        return str(uuid.uuid4())

    def get_user_history_path(self, user_id):
        """Get the path to a user's chat history file"""
        return CHAT_HISTORY_DIR / f"user_{user_id}.json"

    def load_user_history(self, user_id):
        """Load chat history for a specific user"""
        history_file = self.get_user_history_path(user_id)
        if not history_file.exists():
            return None

        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading user history: {e}")
            return None

    def save_user_history(self, user_id, history):
        """Save chat history for a specific user"""
        history_file = self.get_user_history_path(user_id)
        try:
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving user history: {e}")
            return False

    def create_new_user(self):
        """Create a new user with a unique ID"""
        user_id = self.generate_user_id()
        self.current_user = {
            'user_id': user_id,
            'created_at': str(datetime.now()),
            'chat_history': []
        }
        self.save_user_history(user_id, self.current_user)
        return user_id

    def initialize_conversation(self, user_id):
        """Initialize or load conversation for a user"""
        user_data = self.load_user_history(user_id)
        if not user_data:
            return False
            
        self.current_user = user_data
        
        # Create or get message history for this user
        if user_id not in self.histories:
            self.histories[user_id] = ChatMessageHistory()
            
            # Add system message as first message
            self.histories[user_id].add_message(SystemMessage(content=SYSTEM_PROMPT))
            
            # Load previous messages
            for msg in user_data.get('chat_history', []):
                if msg['role'] == 'user':
                    self.histories[user_id].add_message(HumanMessage(content=msg['content']))
                else:
                    self.histories[user_id].add_message(AIMessage(content=msg['content']))
        
        # Create the conversation chain with message history
        self.conversation_chain = RunnableWithMessageHistory(
            self.chain,
            lambda session_id: self.histories.get(user_id, ChatMessageHistory()),
            input_messages_key="input",
            history_messages_key="history"
        )
        
        return True

    def add_to_history(self, role, content):
        """Add a message to the current user's history and track emotions"""
        if not self.current_user:
            return False

        timestamp = str(datetime.now())
        user_id = self.current_user['user_id']
        
        # Track emotions and check for crisis
        if role == "user":
            emotion_result = emotion_detector.detect_emotion(content, user_id)
            emotion_data = {
                'emotions': emotion_result['emotions'],
                'dominant_emotion': emotion_result['dominant_emotion'],
                'crisis_detected': emotion_result['crisis_info']
            }
            
            # If crisis detected, add a special marker
            if emotion_result['crisis_info']['is_crisis']:
                self.current_user['needs_immediate_attention'] = True
        else:
            emotion_data = {}
        
        # Add message to history with emotion data
        message = {
            'role': role,
            'content': content,
            'timestamp': timestamp,
            **emotion_data
        }
        
        self.current_user['chat_history'].append(message)
        
        # Save the updated history
        return self.save_user_history(user_id, self.current_user)

    # Emotion history is now managed directly in the emotion detector

    def get_emotion_summary(self, user_id: str) -> Dict[str, Any]:
        """Get a summary of user's emotional state"""
        history = emotion_detector.get_emotion_history(user_id)
        if not history:
            return {"status": "No emotion data available"}
        
        # Calculate average emotion scores
        emotion_totals = {}
        for entry in history:
            for emotion, score in entry['emotions'].items():
                if emotion not in emotion_totals:
                    emotion_totals[emotion] = 0
                emotion_totals[emotion] += score
        
        avg_emotions = {e: s/len(history) for e, s in emotion_totals.items()}
        
        # Convert scores to percentages, ensuring they're numbers first
        emotion_percentages = {}
        for k, v in avg_emotions.items():
            try:
                # Ensure the value is a number that can be formatted
                score = float(v) * 100
                emotion_percentages[k] = f"{score:.1f}%"
            except (ValueError, TypeError):
                # If conversion fails, use the raw value
                emotion_percentages[k] = f"{v}"
                
        return {
            'emotion_summary': emotion_percentages,
            'dominant_emotion': max(avg_emotions.items(), key=lambda x: x[1])[0] if avg_emotions else 'neutral',
            'total_messages_analyzed': len(history)
        }

    def _get_emotional_context(self, user_id: str) -> str:
        """Generate context about user's emotional state for the AI."""
        if not hasattr(self, 'emotion_history') or not emotion_detector.emotion_history.get(user_id):
            return ""
            
        # Get recent emotions (last 5 messages)
        recent_emotions = emotion_detector.emotion_history[user_id][-5:]
        if not recent_emotions:
            return ""
            
        # Count occurrences of each emotion
        emotion_counts = {}
        for entry in recent_emotions:
            emotion = entry.get('dominant_emotion', 'neutral')
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
        # Generate context string
        context_parts = ["Recent emotional patterns:"]
        for emotion, count in emotion_counts.items():
            context_parts.append(f"- {emotion.capitalize()}: {count} time{'s' if count > 1 else ''}")
            
        return "\n".join(context_parts)

    def _detect_crisis_keywords(self, text: str) -> bool:
        """Check for crisis-related keywords in the text."""
        crisis_keywords = [
            'suicide', 'end my life', 'kill myself', 'want to die',
            'no reason to live', 'self harm', 'hurting myself',
            "can't take it", 'giving up', 'hopeless', 'helpless'
        ]
        return any(keyword in text.lower() for keyword in crisis_keywords)

    def _get_conversation_context(self, user_id: str) -> str:
        """
        Generate comprehensive context about the conversation history and emotional state.
        
        Returns a structured string containing:
        1. Current emotional state and trends
        2. Summary of recent conversation
        3. Important topics and concerns
        4. Recent messages with emotional context
        """
        # Get recent messages (last 10 messages for better context)
        recent_messages = self.current_user.get('chat_history', [])[-10:]
        
        # Get enhanced emotion analysis
        emotion_summary = self.get_emotion_summary(user_id)
        
        context_parts = ["[CONVERSATION CONTEXT]"]
        
        # 1. Emotional Context
        if emotion_summary.get('dominant_emotion'):
            emotion = emotion_summary.get('dominant_emotion', 'neutral')
            intensity = emotion_summary.get('intensity', 'medium') if emotion != 'neutral' else 'neutral'
            
            # Add emotional state with intensity
            context_parts.append(
                f"Emotional State: {emotion.capitalize()} "
                f"(Intensity: {intensity.capitalize()})"
            )
            
            # Add emotional trend if available
            if emotion_summary.get('trend') and emotion_summary['trend'] != 'stable':
                trend = emotion_summary['trend']
                context_parts.append(
                    f"Emotional Trend: The user has been {trend}ly {emotion} in recent messages."
                )
            
            # Add top emotions if available
            if emotion_summary.get('emotion_summary'):
                top_emotions = sorted(
                    emotion_summary['emotion_summary'].items(),
                    key=lambda x: float(x[1].rstrip('%')),
                    reverse=True
                )[:3]
                
                if top_emotions:
                    context_parts.append(
                        "Recent Emotional Mix: " + 
                        ", ".join(f"{e[0].capitalize()} ({e[1]})" for e in top_emotions)
                    )
        
        # 2. Conversation Summary
        if recent_messages:
            # Identify key topics or themes from recent messages
            topics = self._identify_conversation_topics(recent_messages)
            if topics:
                context_parts.append(
                    "\nKey Topics Discussed: " + 
                    ", ".join(f"{t.capitalize()}" for t in topics)
                )
            
            # Add recent conversation context
            context_parts.append("\n[RECENT MESSAGES]")
            for msg in recent_messages[-5:]:  # Show last 5 messages for context
                role = "You" if msg['role'] == 'user' else "Aanya"
                timestamp = msg.get('timestamp', '')
                
                # Add emotion info if available
                emotion_info = ''
                if msg['role'] == 'user' and 'dominant_emotion' in msg:
                    emotion_info = f" [Felt: {msg['dominant_emotion'].capitalize()}]"
                
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        time_str = dt.strftime('%I:%M %p')
                        context_parts.append(f"{time_str} - {role}: {msg['content']}{emotion_info}")
                        continue
                    except (ValueError, AttributeError):
                        pass
                context_parts.append(f"{role}: {msg['content']}{emotion_info}")
        
        # 3. Previous Concerns or Issues
        if self.current_user.get('needs_follow_up', False):
            context_parts.append(
                "\n[IMPORTANT] This user was previously in crisis and may need follow-up care. "
                "Be especially attentive to their emotional state and needs."
            )
        
        return "\n".join(context_parts)
        
    def _identify_conversation_topics(self, messages: List[Dict]) -> List[str]:
        """
        Identify key topics or themes from recent messages.
        This is a simple implementation that can be enhanced with NLP.
        """
        # Simple keyword-based topic detection
        topic_keywords = {
            'studies': ['exam', 'test', 'study', 'homework', 'assignment', 'class', 'school', 'college', 'marks'],
            'family': ['mom', 'dad', 'parents', 'family', 'sister', 'brother', 'mother', 'father'],
            'relationships': ['friend', 'girlfriend', 'boyfriend', 'partner', 'relationship', 'dating'],
            'career': ['job', 'career', 'future', 'interview', 'resume', 'placement', 'internship'],
            'stress': ['stress', 'anxious', 'anxiety', 'worried', 'overwhelmed', 'pressure'],
            'sleep': ['sleep', 'tired', 'insomnia', 'can\'t sleep', 'restless'],
            'health': ['sick', 'ill', 'pain', 'headache', 'stomach', 'doctor', 'hospital']
        }
        
        # Count occurrences of each topic
        topic_counts = {topic: 0 for topic in topic_keywords}
        
        for msg in messages:
            content = msg.get('content', '').lower()
            for topic, keywords in topic_keywords.items():
                if any(keyword in content for keyword in keywords):
                    topic_counts[topic] += 1
        
        # Return topics that were mentioned at least twice
        return [topic for topic, count in topic_counts.items() if count >= 2]

    def get_response(self, user_input: str) -> str:
        """Get response from the AI model with enhanced emotional intelligence and crisis handling."""
        if not self.current_user or not self.conversation_chain:
            return "Error: No active conversation. Please start or load a chat first."
            
        try:
            user_id = self.current_user['user_id']
            
            # Check for crisis first (before adding to history to avoid saving crisis messages)
            crisis_result = self.crisis_detector.detect_crisis(user_input)
            
            # Add user message to history
            self.add_to_history("user", user_input)
            
            # Get conversation context
            context_str = self._get_conversation_context(user_id)
            
            # Format the user input with context
            formatted_input = f"{context_str}\n\nUser: {user_input}"
            
            # Get AI response using the conversation chain
            response = self.conversation_chain.invoke(
                {"input": formatted_input},
                {"configurable": {"session_id": user_id}}
            )
            
            # Extract the response content
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Handle crisis situations with appropriate escalation
            if crisis_result['is_crisis']:
                # Format crisis response with emergency contacts and support
                crisis_response = (
                    "🚨 [URGENT] 🚨\n"
                    "I'm really concerned about what you're sharing. Your safety is the most important thing right now.\n\n"
                    "Please reach out to these 24/7 helplines immediately:\n"
                    "• Vandrevala Foundation: 1860-2662-345 or 1800-2333-330 (24/7, free from all phones)\n"
                    "• iCall: +91-9152987821 (Mon-Sat, 10am-8pm, WhatsApp available)\n"
                    "• AASRA: +91-9820466726 (24/7, English/Hindi)\n\n"
                    "You don't have to go through this alone. These trained counselors can help.\n\n"
                    f"{response_text}"
                )
                response_text = crisis_response
                
                # Set a flag for follow-up in future sessions
                self.current_user['needs_follow_up'] = True
                
            elif crisis_result['risk_level'] == 'medium':
                # For medium risk, show concern and offer resources
                response_text = (
                    "🤗 I hear how much you're struggling right now, and I want you to know I'm here for you.\n\n"
                    "Sometimes talking to someone can help. These free, confidential services are available:\n"
                    "• Vandrevala: 1860-2662-345 (24/7)\n"
                    "• iCall: 9152987821 (Mon-Sat, 10am-8pm)\n\n"
                    f"{response_text}"
                )
            
            # Add to history and save
            self.add_to_history("assistant", response_text)
            self.save_user_history(user_id, self.current_user)
            
            return response_text
            
        except Exception as e:
            print(f"Error in get_response: {str(e)}")
            return "I'm sorry, I'm having trouble processing that right now. Could you try again?"

    def get_emotion_summary(self, user_id: str) -> Dict[str, Any]:
        """Get a summary of user's emotional state"""
        history = emotion_detector.get_emotion_history(user_id)
        if not history:
            return {"status": "No emotion data available"}
        
        # Calculate average emotion scores
        emotion_totals = {}
        for entry in history:
            for emotion, score in entry['emotions'].items():
                if emotion not in emotion_totals:
                    emotion_totals[emotion] = 0
                emotion_totals[emotion] += score
        
        avg_emotions = {e: s/len(history) for e, s in emotion_totals.items()}
        
        # Convert scores to percentages, ensuring they're numbers first
        emotion_percentages = {}
        for k, v in avg_emotions.items():
            try:
                # Ensure the value is a number that can be formatted
                score = float(v) * 100
                emotion_percentages[k] = f"{score:.1f}%"
            except (ValueError, TypeError):
                # If conversion fails, use the raw value
                emotion_percentages[k] = f"{v}"
                
        return {
            'emotion_summary': emotion_percentages,
            'dominant_emotion': max(avg_emotions.items(), key=lambda x: x[1])[0] if avg_emotions else 'neutral',
            'total_messages_analyzed': len(history)
        }

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the application header"""
    clear_screen()
    print("\n" + "="*50)
    print("Welcome to Mental Health Chatbot".center(50))
    print("Type 'exit' to quit".center(50))
    print("="*50 + "\n")

def main():
    chat_manager = UserChatManager()
    print_header()

    # Main menu loop
    while True:
        print("\nAre you a new or existing user?")
        print("1. New User")
        print("2. Existing User")
        print("3. Exit")
        choice = input("\nEnter your choice (1-3): ").strip()

        if choice == '1':
            # New User
            user_id = chat_manager.create_new_user()
            print(f"\nWelcome! Your User ID is: {user_id}")
            print("Please save this ID for future reference.")
            if chat_manager.initialize_conversation(user_id):
                break

        elif choice == '2':
            # Existing User
            user_id = input("\nEnter your User ID: ").strip()
            if chat_manager.initialize_conversation(user_id):
                print(f"\nWelcome back, User {user_id}!")
                break
            else:
                print("User not found. Please try again or create a new user.")

        elif choice == '3':
            print("\nGoodbye!")
            return
        else:
            print("Invalid choice. Please try again.")

    # Main chat loop
    print("\nChat started. Type 'exit' to end the session.\n")
    try:
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'exit':
                    if chat_manager.current_user:
                        try:
                            print("\nEmotion Summary:")
                            summary = chat_manager.get_emotion_summary(chat_manager.current_user['user_id'])
                            if 'emotion_summary' in summary:
                                for emotion, score in summary['emotion_summary'].items():
                                    print(f"{emotion.capitalize()}: {score}")
                                print(f"\nDominant emotion: {summary.get('dominant_emotion', 'N/A')}")
                        except Exception as e:
                            print(f"\nCouldn't generate emotion summary: {str(e)}")
                    print("\nSaving your chat history. Goodbye!")
                    break
                    
                if user_input.lower() == 'emotions':
                    if chat_manager.current_user:
                        summary = chat_manager.get_emotion_summary(chat_manager.current_user['user_id'])
                        print("\nEmotion Analysis:")
                        for emotion, score in summary.get('emotion_summary', {}).items():
                            print(f"{emotion.capitalize()}: {score*100:.1f}%")
                        print(f"\nDominant emotion: {summary.get('dominant_emotion', 'N/A')}")
                    continue

                # Get response from the AI model with emotional intelligence
                response = chat_manager.get_response(user_input)
                print(f"\nAssistant: {response}\n")

            except KeyboardInterrupt:
                # Allow Ctrl+C to exit cleanly
                raise
            except Exception as e:
                print(f"\nSorry, I encountered an error: {str(e)}")
                print("Please try again or type 'exit' to quit.\n")

    except KeyboardInterrupt:
        print("\n\nSession interrupted. Saving your chat history...")
    finally:
        if chat_manager.current_user:
            chat_manager.save_user_history(
                chat_manager.current_user['user_id'],
                chat_manager.current_user
            )

if __name__ == "__main__":
    main()