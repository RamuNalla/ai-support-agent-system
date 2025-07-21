# Use an official Python runtime as a parent image
FROM python

WORKDIR /app

COPY agent_service/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY agent_service/app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]