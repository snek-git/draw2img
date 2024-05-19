const colorInput = document.getElementById('color');
const weight = document.getElementById('weight');
const clear = document.getElementById('clear');
const promptInput = document.getElementById('prompt');
const paths = [];
let currentPath = [];

function setup() {
    createCanvas(512, 512).parent('image-container');
    background(255);
}

function draw() {
    noFill();
    if (mouseIsPressed) {
        const point = { x: mouseX, y: mouseY, color: colorInput.value, weight: weight.value };
        currentPath.push(point);
    }
    paths.forEach(path => {
        beginShape();
        path.forEach(point => {
            stroke(point.color);
            strokeWeight(point.weight);
            vertex(point.x, point.y);
        });
        endShape();
    });
}

function mousePressed() {
    currentPath = [];
    paths.push(currentPath);
}

function mouseReleased() {
    if (isMouseInsideCanvas()) {
        sendImage();  // Send image when mouse is released
    }
}

function keyPressed() {
    if (key === '[') {
        let newSize = max(1, int(weight.value) - 1); // Decrease stroke size, minimum 1
        weight.value = newSize;
    } else if (key === ']') {
        let newSize = min(200, int(weight.value) + 1); // Increase stroke size, maximum 200
        weight.value = newSize;
    }
}

function isMouseInsideCanvas() {
    return mouseX >= 0 && mouseX <= width && mouseY >= 0 && mouseY <= height;
}

clear.addEventListener('click', () => {
    paths.splice(0);
    background(255);
});

function sendImage() {
    saveFrames('out', 'png', 1, 1, function(im) {
        const prompt = promptInput.value || "flock of birds flying over the sea, splash art";
        console.log("Sending image data:", im[0].imageData);  // Log the image data being sent
        axios.post('http://localhost:5000/generate', {
            image: im[0].imageData,
            prompt: prompt, // Add the prompt to the request
            steps: 4,
            cfg_scale: 1,
            strength: 0.9
        }, {
            headers: {
                'Content-Type': 'application/json' // Sending JSON data
            }
        })
        .then(response => {
            displayImageFromBase64(response.data.image); // Use the image data from the response
        })
        .catch(error => {
            console.error('Error sending image:', error);
        });
    });
}

function displayImageFromBase64(base64ImageString) {
    const img = document.createElement('img'); // Create a new <img> element
    img.src = 'data:image/png;base64,' + base64ImageString;  // Set the base64 string as the image source

    // Optional: Add styling or attributes
    img.alt = 'Output image';

    // Append to the desired location in your HTML
    const imageContainer = document.getElementById('result-container');
    imageContainer.innerHTML = '';
    imageContainer.appendChild(img); 
}