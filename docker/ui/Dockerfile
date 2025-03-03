FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir streamlit==1.34.0 requests==2.31.0 pandas==2.1.4

# Copy the Streamlit app
COPY ui/app.py .

# Set environment variables
ENV API_URL=http://api:8000

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
