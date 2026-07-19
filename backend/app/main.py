from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import init_db
from app.routers import auth, collections, questions, solver, upload

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="AI-powered Calculus Solver & Checker with Question Library",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(solver.router)
app.include_router(questions.router)
app.include_router(collections.router)
app.include_router(auth.router)
app.include_router(upload.router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to the Calculus Solver & Checker API",
        "version": app.version,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }


@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": app.version,
    }


@app.get("/api", tags=["System"])
async def api_info():
    return {
        "name": settings.APP_NAME,
        "version": app.version,
        "description": app.description,
        "documentation": "/docs",
        "redoc": "/redoc",
    }