"""Start the local-only server with one process and one worker."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("relaxrouter.api:app", host="127.0.0.1", port=8000, workers=1)
