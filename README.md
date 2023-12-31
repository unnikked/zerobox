# Zerobox

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/downloads/release/python-390/)

## Description
This is a Flask app that provides an API for image zero-shot object detection based on [CLIPSeg](https://huggingface.co/blog/clipseg-zero-shot). It takes in an image URL and a list of labels with thresholds.

The app uses the Flask framework to handle HTTP requests and a machine learning model for predictions. It can be easily integrated into other applications or used as a standalone API.

- [ ] Insert threshold usage explanation

## Features
- Accepts POST requests at `/detext` to detect bounding boxes based on an image URL and provided thresholds.
- Accepts POST requests at `/draw` to draw bounding boxes on an image based on the provided labels and thresholds.

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/unnikked/zerobox.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up any necessary environment variables (if applicable).

## Usage
1. Run the Flask app:
   ```bash
   python app.py
   ```
2. Send a POST request to `http://localhost:5000/detect` or `http://localhost:5000/draw` with the required JSON payload.
3. Retrieve the JSON response or the modified image (depending on the endpoint).

## API Documentation
- **Endpoint:** `/predict`
  - **Method:** POST
    - **Payload:**
      ```json
      {
        "image_url": "https://d.newsweek.com/en/full/1809693/cat-dog.jpg?w=1600&h=1600&q=88&f=117239ddc10e0929372035ef0b425e2e",
        "prompts": ["cat:0.91", "dog:0.9"]
      }
    ```
  - **Response:**
    ```json
    {
      "image_url": "https://d.newsweek.com/en/full/1809693/cat-dog.jpg?w=1600&h=1600&q=88&f=117239ddc10e0929372035ef0b425e2e",
      "predictions": [
        {
            "bbox": {
                "xmax": 1509,
                "xmin": 736,
                "ymax": 1222,
                "ymin": 654
            },
            "label": "cat",
            "threshold": 0.91
        },
        {
            "bbox": {
                "xmax": 795,
                "xmin": 68,
                "ymax": 1181,
                "ymin": 295
            },
            "label": "dog",
            "threshold": 0.9
        }
      ]
    }
    ```

- **Endpoint:** `/draw`
  - **Method:** POST
  - **Payload:**
    ```json
    {
        "image_url": "https://d.newsweek.com/en/full/1809693/cat-dog.jpg?w=1600&h=1600&q=88&f=117239ddc10e0929372035ef0b425e2e",
        "prompts": ["cat:0.91", "dog:0.9"]
    }
    ```
  - **Response:** ![](https://i.imgur.com/TpJY3jP.jpg)

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.