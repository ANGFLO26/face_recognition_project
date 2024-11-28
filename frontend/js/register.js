import { capturePhoto, activateCamera } from "./camera.js";

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', () => {
    const activateButton = document.getElementById('activateButton');
    const captureButton = document.getElementById('captureButton');

    if (activateButton) {
        activateButton.addEventListener('click', activateCamera);
    }

    if (captureButton) {
        captureButton.addEventListener('click', capturePhoto);
    }
});
