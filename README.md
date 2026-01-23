# Deepfake Detection System

A high-accuracy deepfake detection system built with PyTorch and Flask.

## Project Structure

- `src/`: Source code for model definitions and preprocessing.
- `training/`: Jupyter Notebooks for training on Google Colab.
- `backend/`: Flask application for inference.
- `frontend/`: Web interface for users (Vite/React or static HTML).
- `models/`: Trained model weights.

## How to Run Locally

To share this project with a friend, they should follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Bhavani-Bp/deepfakedetectionusing-vit-.git
   cd deepfakedetectionusing-vit-
   ```

2. **Install Dependencies**:
   It's recommended to use a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Verify Model Weights**:
   Ensure the model weight file is present in `models/`. The backend expects `models/deepfake_vit_best.pth`. 
   > [!NOTE]
   > If you have a different weight file (e.g., `deepfake_efficientnet.pth`), you must update the filename in `backend/app.py` at line 40.

4. **Run the Backend**:
   ```bash
   python backend/app.py
   ```
   The server will start at `http://127.0.0.1:5000`.

5. **Open the App**:
   Open a browser and go to `http://127.0.0.1:5000` to use the identification system.

## How to Run on Google Colab

1. **Upload**: Upload this entire project folder to your Google Drive.
2. **Open Notebook**: Navigate to `training/` and open `01_environment_setup.ipynb`.
3. **Mount Drive**: The notebook will guide you to mount your Google Drive.
4. **Install Dependencies**: Run the cell to install `requirements.txt`.
5. **Verify**: Ensure the GPU is active and libraries are working.

## Requirements

See `requirements.txt`. Key dependencies:
- PyTorch
- OpenCV
- Facenet-PyTorch
- Flask
- Flask-CORS
