from fastapi import FastAPI

app = FastAPI()

@app.get("/get")
def get():
    return {}