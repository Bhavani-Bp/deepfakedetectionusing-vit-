# Deepfake Detection System

A high-accuracy deepfake detection system built with PyTorch and FastAPI.

## Project Structure

- `data/`: Raw and processed datasets (Real vs Fake).
- `src/`: Source code for model definitions and preprocessing.
- `training/`: Jupyter Notebooks for training on Google Colab.
- `backend/`: FastAPI application for inference.
- `frontend/`: Web interface for users.

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
