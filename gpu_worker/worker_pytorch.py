from fastapi import FastAPI, File, UploadFile
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn


app = FastAPI()

@app.post('/spellchecker')
def _file_upload(file: UploadFile, file2: UploadFile, delimiter: str = Form(","), returnType: str = Form("html")):
    data = {'delimiter': delimiter, 'returnType': returnType}
    print(data)
    print(file)
    print(file2)
    # files = {'file': file.file.read(), 'file2': file2.file.read()}
    return JSONResponse(content={'content': 'at the model'})


if __name__ == "__main__":
    uvicorn.run("worker_pytorch:app", host="0.0.0.0", port=8000, log_level="debug")
