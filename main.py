from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.app import api_router


# Define app
app = FastAPI(
    title = "Vector Search Engine Using Faiss",
    desctiption = "Search Most Similar Fashion Item",
    version= "0.1.0",
)

# CORS Middleware Setting
app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Run Router
app.include_router(api_router)