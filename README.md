# Image Captioning Project

This project is an image captioning system that generates captions for images and can translate them into different languages. The project is structured into several components, including training code, backend services, and static files for the web interface.

## Project Structure

### Files and Directories

- **backend.py**: Contains the backend code for the Flask web application, including routes for uploading images and generating captions.
- **caption/**: Directory containing the finetuned model.
- **requirements.txt**: Lists the Python dependencies required for the project.
- **static/**: Directory for static files when user upload the image.
- **templates/**: Directory for HTML templates used by the Flask application.
- **train-code/**: Directory containing the training and evaluation scripts for the image captioning model. (Not using in the API)

## Setup and Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/ModzabazeR/image-caption-generator-backend.git
    cd image-caption-generator-backend
    ```

2. **Install Python dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Download NLTK data**:
    ```sh
    python -m nltk.downloader punkt
    ```

## Training the Model

The training code is located in the `train-code` directory. The main script for training is `train.py`.

## Running the API

To run the API, execute the following command:

```sh
python backend.py
```