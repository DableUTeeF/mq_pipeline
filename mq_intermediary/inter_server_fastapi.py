from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import io
import pika
import os
import uuid
import time


app = FastAPI()


class RPCClient(object):
    def __init__(self):
        url = os.environ.get('CLOUDAMQP_URL', 'amqp://guest:guest@sr-mq/%2f')
        params = pika.URLParameters(url)
        self.connection = pika.BlockingConnection(params)

        self.channel = self.connection.channel()

        result = self.channel.queue_declare('', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.on_response,
            auto_ack=True)

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, body):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(
            exchange='',
            routing_key='capgen_queue',
            properties=pika.BasicProperties(
                reply_to=self.callback_queue,
                correlation_id=self.corr_id,
            ),
            body=body)
        while self.response is None:
            self.connection.process_data_events()
        return self.response


def process_img(cvimg, org_filename=''):
    client = RPCClient()
    result = client.call(cvimg)
    return result


def on_request(ch, method, props, body):
    response = requests.post(url, **body)
    if isinstance(response, dict):
        response = json.dumps(response)
    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id=props.correlation_id),
                     body=response)
    ch.basic_ack(delivery_tag=method.delivery_tag)


if __name__ == "__main__":
    url = os.environ.get('CLOUDAMQP_URL', 'amqp://guest:guest@sr-mq/%2f')  # Taz check!
    time.sleep(5)
    params = pika.URLParameters(url)
    connection = pika.BlockingConnection(params)

    channel = connection.channel()
    channel.queue_declare(queue='capgen_queue')

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='capgen_queue', on_message_callback=on_request)

    uvicorn.run("inter_server_fastapi:app", host="0.0.0.0", port=8000, log_level="debug")
