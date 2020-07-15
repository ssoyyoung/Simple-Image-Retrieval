from pydantic import BaseModel

class Input(BaseModel):
    b64Image: str