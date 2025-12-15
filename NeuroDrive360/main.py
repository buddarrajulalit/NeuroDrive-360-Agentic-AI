"""Main FastAPI application entry point.

This module initializes and configures the FastAPI application for the
automotive predictive maintenance system.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

# Initialize FastAPI application
app = FastAPI(
    title="Automotive Predictive Maintenance System",
    description=(
        "A FastAPI-based system for predictive maintenance of vehicles using "
        "AI-powered diagnosis and machine learning models."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint.

    Returns:
        Dictionary with API information.
    """
    return {
        "message": "Automotive Predictive Maintenance System API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

