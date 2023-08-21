# Base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy requirements.txt to the working directory
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app folder to the working directory
COPY . .

RUN python download.py

# Expose the port that the Flask app will run on
EXPOSE 8080

# Set environment variables
ENV FLASK_APP=main.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the Flask application
CMD ["flask", "run"]