"""Run API locally for testing"""
import uvicorn
import os
import sys

if __name__ == "__main__":
    # Allow port to be set via environment variable or command line arg
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}, using default 8000")
    else:
        port = int(os.getenv("API_PORT", "8000"))
    
    print(f"Starting Predictr API on http://localhost:{port}")
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )

