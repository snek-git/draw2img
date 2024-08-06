# draw2img

- Canvas for drawing sketches
- Real-time image generation using Stable Diffusion 1.5 with LCM LoRA
- Customizable brush color and size
- Adjustable generation parameters (steps, CFG scale, strength)

## Setup

### Backend

1. Install required Python dependencies:
   ```
   pip install torch diffusers transformers flask flask-cors gradio pillow
   ```

2. Set up the models directory:
   ```
   mkdir models
   ```
   This directory will be used to cache downloaded AI models, speeding up subsequent runs and allowing potential offline use.

3. Run the Flask server:
   ```
   python app.py
   ```

### Frontend

1. Ensure you have Node.js and npm installed on your system.

2. Navigate to the frontend directory:
   ```
   cd frontend
   ```

3. Initialize a new npm project (if not already done):
   ```
   npm init -y
   ```

4. Install required dependencies:
   ```
   npm install http-server
   ```

5. In a new terminal, run:
   ```
   http-server -p 8080
   ```

## Running the Application

1. Start the backend server:
   ```
   python app.py
   ```

2. In a new terminal, start the frontend server:
   ```
   cd frontend
   npm start
   ```

3. Open a web browser and navigate to `http://localhost:8080` to use the application.

## Usage

1. Draw a sketch on the canvas.
2. Enter a text prompt describing the desired image.
3. Adjust generation parameters if needed.
4. Release the mouse to trigger image generation.
5. The AI-generated image will appear next to your sketch.

## Backend Options

- Use `--gradio` flag to run both Flask and Gradio interfaces.
- Use `--share` for public Gradio URL (when using Gradio).
- Use `--port` to specify a custom port for the Flask server.

## Note on Model Caching

The `models` directory is used to cache downloaded AI models. This improves performance by:
- Speeding up subsequent runs by avoiding re-downloads
- Centralizing model storage to save disk space
- Enabling potential offline use once models are cached

First-time runs may take longer as models are downloaded and cached.

## Troubleshooting

If you encounter CORS issues, ensure that your Flask backend has CORS properly configured. You may need to adjust the allowed origins in the Flask app if you're running the frontend on a different port or domain.
