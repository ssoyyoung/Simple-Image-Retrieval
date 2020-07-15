import io
import base64
from io import BytesIO

def decode_b64_PIL(base64_string):
    b64 = base64.b64decode(base64_string)
    bufImg = io.BytesIO(b64)

    return bufImg

def decode_b64_to_pil(base64_string):
    f = BytesIO()
    f.write(base64.b64decode(base64_string))

    return f
