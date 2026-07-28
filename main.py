import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import PORT
from routers import extract

app = FastAPI(
    title="Food Recipe AI Service",
    description="AI microservice for batch recipe extraction from YouTube video transcripts.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(extract.router)

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "food-recipe-ai-service"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)
