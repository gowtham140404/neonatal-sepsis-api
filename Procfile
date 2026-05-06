# Procfile — used by Render, Heroku, Railway, and similar platforms
# Uses gunicorn with uvicorn workers for production (not uvicorn alone)
# --workers 2: safe default; set to (2 * CPU_cores + 1) for more throughput
# --worker-class uvicorn.workers.UvicornWorker: async support
# --timeout 120: 2 min timeout for slow cold starts (model loading)
# --bind 0.0.0.0:$PORT: Render injects $PORT automatically

web: gunicorn main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --timeout 120 --bind 0.0.0.0:$PORT
