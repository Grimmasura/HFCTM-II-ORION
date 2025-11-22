FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY orion_api.py .
CMD ["uvicorn", "orion_api:app", "--host", "0.0.0.0", "--port", "8080"]