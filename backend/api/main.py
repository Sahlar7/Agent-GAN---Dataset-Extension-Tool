from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import jobs_router
from api.config import CORS_ORIGINS

app = FastAPI(title="Datagent API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(jobs_router)

@app.get("/")
async def root():
    return {"message": "Datagent API", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)