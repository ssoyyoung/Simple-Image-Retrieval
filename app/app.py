from fastapi import APIRouter
from app.api.endPoint import createFaissIndex, createElasticIndex
from app.api.endPoint import searchFaissIndex, searchElasticIndex

api_router = APIRouter()

api_router.include_router(createFaissIndex.router, prefix="/faiss/create")
api_router.include_router(createElasticIndex.router, prefix="/elastic/create")

api_router.include_router(searchFaissIndex.router, prefix="/faiss/search")
# api_router.include_router(createElasticIndex.router, prefix="/elastic/search")