import pika
import os
import uuid
import time
import requests
import json
from io import BytesIO


def on_request(ch, method, props, body):
    print(props, flush=True)
    body = json.loads(body)
    print(body, flush=True)
    url = body.pop('url')
    response = requests.post(
        url,
        **body,
        # headers=props.headers
    )
    print(response)
    if isinstance(response, dict):
        response = json.dumps(response)
    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id=props.correlation_id),
                     body=response.content)
    ch.basic_ack(delivery_tag=method.delivery_tag)


if __name__ == "__main__":
    url = os.environ.get('CLOUDAMQP_URL', 'amqp://guest:guest@sr-mq/%2f')
    time.sleep(5)
    params = pika.URLParameters(url)
    connection = pika.BlockingConnection(params)

    channel = connection.channel()
    channel.queue_declare(queue='capgen_queue')

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='capgen_queue', on_message_callback=on_request)
    print(" [x] Awaiting RPC requests")
    channel.start_consuming()
