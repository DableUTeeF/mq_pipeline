from fastapi import FastAPI, File, UploadFile
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import io
import pika
import os
import uuid


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


@app.post('/spellchecker')
def _file_upload(file: UploadFile, file2: UploadFile, delimiter: str = Form(","), returnType: str = Form("html")):
    data = {'delimiter': delimiter, returnType: returnType}
    files = {'file': file.file.read(), 'file2': file2.file.read()}
    client = RPCClient()
    result = client.call({'data': data, 'files': files})
    return JSONResponse(content=result)


if __name__ == "__main__":
    uvicorn.run("public_server_fastapi:app", host="0.0.0.0", port=8000, log_level="debug")
