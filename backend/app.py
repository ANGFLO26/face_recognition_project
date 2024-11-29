from flask import Flask, send_from_directory, jsonify, request , render_template
from model import FaceRecognitionModel
from pymongo.errors import ConnectionFailure
import base64
from io import BytesIO
from PIL import Image

# Create the Flask app and define static files path
app = Flask(__name__, static_folder="/home/phanvantai/Documents/THREE_YEARS/MACHINE_LEARNING/THE_END/frontend", static_url_path="")


# Initialize the face recognition model
model = FaceRecognitionModel()

# Serve the index page
@app.route("/")
def serve_index():
    model.retrain_model()
    return send_from_directory(app.static_folder, "index.html")

@app.route("/main")
def serve_main():
    return send_from_directory(app.static_folder, "main.html")

# Serve the login page
@app.route("/login")
def serve_login():
    return render_template('login.html')

# Serve the register page
@app.route("/register")
def serve_register():
    return send_from_directory(app.static_folder, "register.html")

# Serve static files (CSS, JS)
@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)

# Check MongoDB connection
def check_mongo_connection():
    try:
        model.client.admin.command("ping")
        print("✅ MongoDB connection successful!")
    except ConnectionFailure:
        print("❌ MongoDB connection failed. Please check your database connection!")
        return False
    return True

# Registration API
@app.route("/api/register", methods=["POST"])
def register():
    data = request.json
    username = data.get("name")
    image_data = data.get("image")
    # Kiểm tra nếu tên người dùng hoặc dữ liệu ảnh bị thiếu
    if not username or not image_data:
        return jsonify({"error": "Username or image data is missing"}), 400

    
    if image_data:
        try:
            image_data = image_data.split(",")[1]  
        except Exception as e:
            return jsonify({"message": "Invalid image data"}), 400
    else:
        return jsonify({"message": "No image data received"}), 400
    print(f'{username} added for tranining')

   

    # Perform registration
    if model.register_user(username, image_data):
        
       
        return jsonify({"message": "Registration successful"}), 200
    
        
    else:
        return jsonify({"message": "Registration failed"}), 400

# Login API
@app.route("/api/login", methods=["POST"])
def login():
    data = request.json
    image_data = data.get("image")



    # Handle base64 image data
    if image_data:
        try:
            # Decode base64 image data and process it
            image_data = image_data.split(",")[1]
              # Remove base64 header if present
        except Exception as e:
            print(f"Error processing image: {e}")
            return jsonify({"message": "Invalid image data"}), 400
    else:
        return jsonify({"message": "No image data received"}), 400
    try:
        # Perform authentication
        username = model.authenticate_user(image_data)

        
        if username:
            return jsonify({"message": "Login successfully", "username": username}), 200
        else:
            return jsonify({"message": "Login failed. User not recognized"}), 401
    except Exception as e:
        print(f"Error during authentication: {e}")
        return jsonify({"message": "Authentication failed due to internal error."}), 500

# Check MongoDB connection before app starts
if not check_mongo_connection():
    exit()  # Exit if MongoDB connection fails

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)


