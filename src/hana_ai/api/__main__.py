"""
Entry point for running the HANA AI Toolkit API server.
"""
import uvicorn
import logging
from .config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def main():
    """Run the API server."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting HANA AI Toolkit API server on {settings.API_HOST}:{settings.API_PORT}")
    
    uvicorn.run(
        "hana_ai.api.app:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEVELOPMENT_MODE,
        log_level=settings.LOG_LEVEL.lower()
    )

if __name__ == "__main__":
    main()