import requests
from io import BytesIO
from PIL import Image
import base64


# url = "https://api.aiforthai.in.th/sr/sr"
url = "http://0.0.0.0:7620/spellchecker"
files = {'file': open('client_test.py', 'rb'), 'file2': open('docker-compose.yml', 'rb')}
response = requests.post(url, files=files)
# byte_string = bytearray(response.content)
# byte_string = BytesIO(byte_string)
# ori_image = Image.open(byte_string)
# image_string = base64.b64encode(response.content)
print()
