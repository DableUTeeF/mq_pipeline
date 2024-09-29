FROM python:3.9
WORKDIR /app
RUN pip install pika
RUN pip install requests
COPY . .
CMD 'python inter_server_fastapi.py'
