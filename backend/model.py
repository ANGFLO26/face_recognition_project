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
        # Initialize MongoDB connection
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
        self.knn_model = KNeighborsClassifier(n_neighbors=3)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Initialize Mediapipe face detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

        # Ensure folders for temporary image storage
        os.makedirs("temp/login", exist_ok=True)
        os.makedirs("temp/register", exist_ok=True)

        # Set similarity threshold for authentication
        self.similarity_threshold = 0.9

    def save_image(self, image, folder):
        """Save an image to the specified folder and schedule deletion after 5 minutes."""
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(folder, filename)

        image.save(filepath)
        logger.info(f"Image saved to {filepath}")

        # Schedule file deletion after 5 minutes
        Timer(300, self.delete_file, [filepath]).start()

    def delete_file(self, filepath):
        """Delete a file from the system."""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"File {filepath} deleted successfully.")
            else:
                logger.warning(f"File {filepath} not found for deletion.")
        except Exception as e:
            logger.error(f"Error deleting file {filepath}: {e}")

    def extract_face_vector(self, image_data, mode):
        """Extract a face vector using Mediapipe and CLIP."""
        try:
            # Decode base64 image data
            image_data = base64.b64decode(image_data)

            # Convert bytes to a PIL image
            image = Image.open(io.BytesIO(image_data))

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Convert PIL image to numpy array for Mediapipe
            image_np = np.array(image)

            # Perform face detection using Mediapipe
            results = self.face_detection.process(image_np)

            if not results.detections:
                logger.warning("No face detected.")
                return None

            # Extract the first detected face
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box

            # Calculate pixel coordinates for the bounding box
            h, w, _ = image_np.shape
            x_min = int(bbox.xmin * w)
            y_min = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Crop the face region
            face_image = image_np[y_min:y_min + height, x_min:x_min + width]

            # Convert the cropped face to a PIL image
            face_image_pil = Image.fromarray(face_image)

            # Save the cropped face
            folder = "temp/register" if mode == "register" else "temp/login"
            self.save_image(face_image_pil, folder)

            # Extract features using CLIP
            inputs = self.processor(images=face_image_pil, return_tensors="pt")
            outputs = self.clip_model.get_image_features(**inputs)

            logger.info("Face vector extracted successfully.")
            return outputs.detach().numpy().flatten().astype(np.float64)
        except Exception as e:
            logger.error(f"Error extracting face vector: {e}")
            return None

    def majority_vote(self, logreg_pred, knn_pred):
        """Determine the final prediction based on majority voting."""
        return logreg_pred if logreg_pred == knn_pred else logreg_pred

    def register_user(self, username, image_data):
        """Register a new user, ensuring no duplicate username or face."""
        try:
                logger.info(f"Registering user: {username}...")
            # Extract face vector
                face_vector = self.extract_face_vector(image_data, mode="register")
                if face_vector is None:
                    logger.error("Failed to extract face vector.")
                    return False

                # Check for duplicates in database
                users = list(self.users.find())
                for user in users:
                    stored_vector = np.array(user["vector"])
                    similarity = cosine_similarity([face_vector], [stored_vector])[0][0]
                    logger.info(f"Similarity with {user['username']}: {similarity}")

                    if user["username"] == username:
                        logger.error(f"Username '{username}' is already registered.")
                        return False

                    if similarity >= self.similarity_threshold:
                        logger.error(f"Face already registered for username '{user['username']}'.")
                        return False

                # Save new user to database
                self.users.insert_one({"username": username, "vector": face_vector.tolist()})
                logger.info(f"User '{username}' registered successfully.")

                # Update configuration and retrain model if number of users > 2
                self.update_label_mapping()
                self.retrain_model()
                return True
        except Exception as e:
            logger.error(f"Failed to register user. {e}")


    def update_label_mapping(self):
        """Update label mapping in the configuration database."""
        users = list(self.users.find({}, {"username": 1}))
        label_mapping = {user["username"]: idx for idx, user in enumerate(users)}
        self.config.update_one({}, {"$set": {"label_mapping": label_mapping, "class_count": len(label_mapping)}})
        logger.info(f"Updated label mapping: {label_mapping}")

    def retrain_model(self):
        """Retrain the Logistic Regression and KNN models if there are more than 2 users."""
        
        users = list(self.users.find())
        # print('user',users)
        if len(users) < 1:
            logger.info("Don't train because len user < 2")
            return  # Don't train if there are fewer than 3 users
        logger.info("Train Data")
        # Prepare training data
        X_train = []
        y_train = []

        label_mapping = self.config.find_one().get("label_mapping", {})
        for user in users:
            face_vector = np.array(user["vector"])
            X_train.append(face_vector)
            y_train.append(label_mapping[user["username"]])

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Train Logistic Regression model
        logger.info("Training Logistic Regression model...")
        self.logreg_model.fit(X_train, y_train)

        # Train KNN model
        logger.info("Training KNN model...")
        self.knn_model.fit(X_train, y_train)

        logger.info("Models trained successfully.")

    def authenticate_user(self, image_data):
        """Authenticate a user using their image data."""
        logger.info("Authenticating user...")
        
        config = self.config.find_one()
        label_mapping = config.get("label_mapping", {})
        if len(label_mapping) < 2:
            logger.error("Not enough registered users for authentication.")
            return None

        # Extract face vector
        face_vector = self.extract_face_vector(image_data, mode="login")
        if face_vector is None:
            logger.error("Failed to extract face vector.")
            return None

        # Predict using models
        logreg_pred = self.logreg_model.predict([face_vector])[0]
        knn_pred = self.knn_model.predict([face_vector])[0]

        # Decide final prediction using majority voting
        final_label = self.majority_vote(logreg_pred, knn_pred)
        for username, label in label_mapping.items():
            if label == final_label:
                logger.info(f"User '{username}' authenticated successfully.")
                return username

        logger.error("Authentication failed. No matching user found.")
        return None