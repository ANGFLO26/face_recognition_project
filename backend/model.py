import os
import base64
import numpy as np
from datetime import datetime
from threading import Timer
from PIL import Image
from pymongo import MongoClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from transformers import CLIPProcessor, CLIPModel
import mediapipe as mp
import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognitionModel:
    def __init__(self):
        # Initialize database connection
        logger.info("Connecting to MongoDB...")
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["face"]
        self.users = self.db["users"]
        self.config = self.db["config"]

        # Initialize or reset configuration
        if self.config.count_documents({}) == 0:
            logger.info("Initializing configuration...")
            self.config.insert_one({"class_count": 0, "label_mapping": {}})

        # Initialize models
        self.logreg_model = LogisticRegression(max_iter=1000)
        self.knn_model = KNeighborsClassifier(n_neighbors=3)  # Default 3 neighbors
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Initialize Mediapipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

        # Ensure folders exist for temporary image storage
        os.makedirs("temp/login", exist_ok=True)
        os.makedirs("temp/register", exist_ok=True)

        # Similarity threshold for authentication
        self.similarity_threshold = 0.8  # Adjusted threshold for authentication

    def save_image(self, image, folder):
        """Save image to the specified folder and schedule deletion after 5 minutes."""
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(folder, filename)

        image.save(filepath)
        logger.info(f"Image saved to {filepath}")

        # Schedule deletion of the file after 5 minutes
        Timer(300, self.delete_file, [filepath]).start()

    def delete_file(self, filepath):
        """Delete file from the system."""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"File {filepath} deleted successfully.")
            else:
                logger.warning(f"File {filepath} not found for deletion.")
        except Exception as e:
            logger.error(f"Error deleting file {filepath}: {e}")

    def extract_face_vector(self, image_data, mode):
        """Extract face vector using Mediapipe and CLIP."""
        try:
            # Decode base64 string into bytes
            image_data = base64.b64decode(image_data)

            # Convert bytes into PIL image
            image = Image.open(io.BytesIO(image_data))

            # Convert to RGB if not already
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Convert PIL image to numpy array for Mediapipe
            image_np = np.array(image)

            # Mediapipe Face Detection
            results = self.face_detection.process(image_np)

            # If no face is detected
            if not results.detections:
                logger.warning("No face detected.")
                return None

            # Take the first detected face
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box

            # Calculate pixel coordinates from the bounding box
            h, w, _ = image_np.shape
            x_min = int(bboxC.xmin * w)
            y_min = int(bboxC.ymin * h)
            width = int(bboxC.width * w)
            height = int(bboxC.height * h)

            # Crop the face region
            face_image = image_np[y_min:y_min + height, x_min:x_min + width]

            # Convert cropped face to PIL image for CLIP
            face_image_pil = Image.fromarray(face_image)

            # Save cropped face for review
            folder = "temp/register" if mode == "register" else "temp/login"
            self.save_image(face_image_pil, folder)

            # Extract face vector using CLIP
            inputs = self.processor(images=face_image_pil, return_tensors="pt")
            outputs = self.clip_model.get_image_features(**inputs)

            logger.info("Face vector extracted successfully.")
            return outputs.detach().numpy().flatten().astype(np.float64)

        except Exception as e:
            logger.error(f"Error extracting face vector: {e}")
            return None

    def majority_vote(self, logreg_pred, knn_pred):
        """Decide the final prediction based on majority vote."""
        if logreg_pred == knn_pred:
            return logreg_pred
        else:
            return logreg_pred  # Or handle disagreement differently

    def register_user(self, username, image_data):
        """Register a new user. Reject if username or face is already registered."""
        logger.info(f"Registering user: {username}...")
        
        # Extract face vector
        logger.info("Extracting face vector...")
        face_vector = self.extract_face_vector(image_data, mode="register")
        if face_vector is None:
            logger.error("Error: Unable to extract face vector.")
            return False

        # Get list of registered users
        users = list(self.users.find())

        # Check for duplicate username or face
        for user in users:
            stored_vector = np.array(user["vector"])
            similarity = cosine_similarity([face_vector], [stored_vector])[0][0]
            logger.info(f"Comparing with {user['username']}, similarity: {similarity}")

            if user["username"] == username:
                logger.error(f"Error: Username '{username}' is already registered.")
                return False

            if similarity >= self.similarity_threshold:
                logger.error(f"Error: Face is already registered under username '{user['username']}'.")
                return False

        # Add new user
        try:
            result = self.users.insert_one({
                "username": username,
                "vector": face_vector.tolist()  # Convert numpy array to list
            })

            if result.acknowledged:
                logger.info(f"User '{username}' successfully registered.")
            else:
                logger.error(f"Failed to save user '{username}' to the database.")
                return False
        except Exception as e:
            logger.error(f"Error saving user to database: {e}")
            return False

        # Update label mapping and retrain model
        self.update_label_mapping()
        self.update_model()
        return True

    def update_label_mapping(self):
        """Update label mapping in the database."""
        users = list(self.users.find({}, {"username": 1}))
        label_mapping = {user["username"]: idx for idx, user in enumerate(users)}
        class_count = len(label_mapping)

        self.config.update_one({}, {"$set": {"label_mapping": label_mapping}})
        self.config.update_one({}, {"$set": {"class_count": class_count}})
        logger.info(f"Updated label mapping: {label_mapping}")

    def update_model(self):
        """Retrain both Logistic Regression and KNN model using data from the database."""
        config = self.config.find_one()
        label_mapping = config.get("label_mapping", {})
        class_count = len(label_mapping)

        if class_count < 2:
            logger.warning("Not enough classes to train the model. Add more users.")
            return

        data = list(self.users.find())
        X = np.array([np.array(item["vector"], dtype=np.float64) for item in data])
        y = [label_mapping[item["username"]] for item in data]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Logistic Regression
        logger.info("Training Logistic Regression model...")
        self.logreg_model.fit(X_train, y_train)

        # Train KNN
        logger.info("Training KNN model...")
        self.knn_model.fit(X_train, y_train)

        # Evaluate models
        logreg_pred = self.logreg_model.predict(X_test)
        knn_pred = self.knn_model.predict(X_test)

        # Voting based on majority rule (Logistic Regression and KNN)
        final_pred = np.array([self.majority_vote(logreg, knn) for logreg, knn in zip(logreg_pred, knn_pred)])

        # Calculate accuracy and other metrics
        accuracy = accuracy_score(y_test, final_pred)
        logger.info(f"Accuracy: {accuracy * 100:.2f}%")

        logger.info("Classification Report:")
        logger.info(classification_report(y_test, final_pred))

        logger.info("Both models have been trained and evaluated successfully!")

    def authenticate_user(self, image_data):
        """Authenticate a user based on the image data."""
        logger.info("Authenticating user...")
        
        config = self.config.find_one()
        label_mapping = config.get("label_mapping", {})
        class_count = len(label_mapping)

        if class_count < 2:
            logger.error("Not enough classes to authenticate. Please register more users.")
            return None

        face_vector = self.extract_face_vector(image_data, mode="login")
        if face_vector is None:
            logger.error("Error: Unable to extract face vector.")
            return None

        users = list(self.users.find())
        X = np.array([np.array(user["vector"], dtype=np.float64) for user in users])
        y = [label_mapping[user["username"]] for user in users]

        # Predict using Logistic Regression and KNN models
        logreg_pred = self.logreg_model.predict([face_vector])
        knn_pred = self.knn_model.predict([face_vector])

        # Majority voting on predictions
        final_pred = self.majority_vote(logreg_pred[0], knn_pred[0])

        if final_pred is None:
            logger.error("Authentication failed.")
            return None

        for username, label in label_mapping.items():
            if label == final_pred:
                logger.info(f"User '{username}' authenticated successfully.")
                return username

        logger.error("Authentication failed: No matching user.")
        return None
