import { capturePhoto_login, activateCamera } from "./camera.js";

let isLoginInProgress = false; // Prevent multiple login attempts

function loginCapturePhoto() {
    capturePhoto_login()
}


// Wait for DOM to be fully loaded and then set up event listeners
document.addEventListener('DOMContentLoaded', () => {
    const activateButton = document.getElementById('activateButton');
    const captureButton_login = document.getElementById('captureButton_login');

    // Event listener to activate the camera
    if (activateButton) {
        activateButton.addEventListener('click', () => {
            activateCamera(); // Calls the activateCamera function from camera.js
        });
    }

    // Event listener to capture photo and send it for login
    if (captureButton_login) {
        captureButton_login.addEventListener('click', loginCapturePhoto); // Calls the loginCapturePhoto function from login.js
    }
});
