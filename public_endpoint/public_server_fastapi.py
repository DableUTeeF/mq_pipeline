from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import json
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

    def close(self):
        self.channel.close()
        self.connection.close()

    def call(self, body, headers):
        print('call', headers)
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


@app.post('/spellchecker')
def _file_upload(file: UploadFile, file2: UploadFile, request: Request, delimiter: str = Form(","), returnType: str = Form("html")):
    data = {'delimiter': delimiter, 'returnType': returnType}
    print('public', data, flush=True)
    files = {'file': file.file.read().decode('utf-8'), 'file2': file2.file.read().decode('utf-8')}
    client = RPCClient()
    result = client.call(json.dumps({'data': data,
                                     'files': files,
                                     'url': 'http://gpu_worker:8000/spellchecker'
                                     }))
    print(result)
    client.close()
    return JSONResponse(content=json.loads(result))


if __name__ == "__main__":
    uvicorn.run("public_server_fastapi:app", host="0.0.0.0", port=8000, log_level="debug")
