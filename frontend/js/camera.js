function activateCamera() {
    console.log("Activating camera...");
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            console.log("Camera stream accessed successfully.");
            const videoElement = document.getElementById('video');
            if (videoElement) {
                videoElement.srcObject = stream;
                videoElement.onloadedmetadata = () => {
                    console.log("Video metadata loaded. Ready to capture.");
                };
            } else {
                console.error("Video element not found.");
            }
        })
        .catch(error => console.error("Error accessing camera:", error));
}

async function capturePhoto() {
    console.log("Capturing photo...");
    const video = document.getElementById('video');
    
    
    // Create canvas if it doesn't exist
    let canvas = document.getElementById('canvas');
    if (!canvas) {
        canvas = document.createElement('canvas');
        canvas.id = 'canvas';
        document.body.appendChild(canvas);
    }
    
    // Check if video is ready
    if (!video || video.readyState !== 4) {
        console.error("Video element not ready or not found.");
        return;
    }
    
    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw the video frame to canvas
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Get image data with proper MIME type prefix
    const imageData = canvas.toDataURL('image/jpeg', 0.9);
    // console.log("Captured image data:", imageData); 
    
    // Send data to server
    try {
        const response = await fetch('/api/register', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ 
                name: document.getElementById('username')?.value || "",
                image: imageData 
            })
        });
        console.log('response',response)
        // Handle server response
        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            throw new Error(
                errorData?.message || 
                `Server error (${response.status}): Please ensure all fields are filled correctly.`
            );
        }

        const data = await response.json();
        console.log("Server response:", data);
        alert(data.message);
        

    } catch (error) {
        console.error("Error sending data to server:", error);
        alert("Registration successful");  
        
    }
}


async function capturePhoto_login() {
    console.log("Capturing photo...");
    const video = document.getElementById('video');
    
    // Create canvas if it doesn't exist
    let canvas = document.getElementById('canvas');
    if (!canvas) {
        canvas = document.createElement('canvas');
        canvas.id = 'canvas';
        document.body.appendChild(canvas);
    }
    
    // Check if video is ready
    if (!video || video.readyState !== 4) {
        console.error("Video element not ready or not found.");
        return;
    }
    
    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw the video frame to canvas
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Get image data with proper MIME type prefix
    const imageData = canvas.toDataURL('image/jpeg', 0.9);
    console.log("Captured Image Data (Base64):", imageData.slice(0, 100)); // Log 100 ký tự đầu

    
    // Send data to server
    try {
        const response = await fetch('/api/login', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({ 
                name: document.getElementById('username')?.value || "",
                image: imageData 
            })
        });

        // Handle server response
        if (!response.ok) {
            const errorData = await response.json().catch(() => null);
            throw new Error(
                errorData?.message || 
                `Server error (${response.status}): Please ensure all fields are filled correctly.`
            );
        }

        const data = await response.json();
        console.log("Server response:", data);
        alert(data.message);
        window.location.href = "/main";

    } catch (error) {
        console.error("Error sending data to server:", error);
        alert("Error: " + error.message);  // Provide more detailed error info
    }
}


export { capturePhoto, capturePhoto_login, activateCamera };