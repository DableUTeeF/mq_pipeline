FROM python:3.9
WORKDIR /app
RUN pip install fastapi
RUN pip install uvicorn
RUN pip install python-multipart
COPY . .
CMD 'python public_server_fastapi.py'
