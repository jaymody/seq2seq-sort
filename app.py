from fastapi import FastAPI

from api import api

app = FastAPI(title="Seq2Seq Sort")
app.include_router(api)
