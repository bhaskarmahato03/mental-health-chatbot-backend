# app.py

from flask import Flask, request, jsonify
from flask_cors import CORS

# Import the main class from your existing chatbot file
from terminal_chat import UserChatManager

# --- Initialize the Application and Chat Manager ---
app = Flask(__name__)
CORS(app)  # This enables cross-origin requests, allowing your frontend to call the API

# Create a single, global instance of the chat manager.
# This loads the models and sets up everything only once when the server starts.
print("Initializing Chat Manager and loading models... Please wait.")
chat_manager = UserChatManager()
print("Initialization complete. Server is ready to accept requests.")
# ----------------------------------------------------

# --- API Endpoints ---

@app.route('/create_user', methods=['POST'])
def create_user():
    """
    Creates a new user and returns their unique ID.
    The frontend should call this when a new user starts a session.
    """
    try:
        user_id = chat_manager.create_new_user()
        if user_id:
            # The frontend needs to save this user_id to use in future chat requests
            return jsonify({'user_id': user_id, 'message': 'New user created successfully.'})
        else:
            return jsonify({'error': 'Failed to create a new user.'}), 500
    except Exception as e:
        print(f"Error in /create_user: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles the main chat interaction.
    Requires a user_id and a message from the frontend.
    """
    data = request.json
    user_id = data.get('user_id')
    user_message = data.get('message')

    if not user_id or not user_message:
        return jsonify({'error': 'Request must include user_id and message.'}), 400

    try:
        # Initialize the conversation for the user. This loads their history.
        if not chat_manager.initialize_conversation(user_id):
            return jsonify({'error': 'User ID not found or invalid.'}), 404

        # Get the chatbot's response using your existing logic
        bot_reply = chat_manager.get_response(user_message)

        return jsonify({'reply': bot_reply})
    except Exception as e:
        print(f"Error in /chat: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500

# This is the standard block to run the Flask application
if __name__ == '__main__':
    # Use host='0.0.0.0' to make the server accessible on your local network
    app.run(host='0.0.0.0', port=5000, debug=False)