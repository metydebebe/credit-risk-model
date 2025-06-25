# Use official Python image as base
FROM python:3.10

# Set working directory inside container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the code
COPY . .

# Default command to run your training script
CMD ["python", "src/train.py"]
