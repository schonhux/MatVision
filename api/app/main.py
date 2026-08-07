from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import auth, matches, jobs, events, annotations, datasets, reports

app = FastAPI(
    title="MatVision API",
    description="AI-powered wrestling film intelligence",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(matches.router)
app.include_router(jobs.router)
app.include_router(events.router)
app.include_router(annotations.router)
app.include_router(datasets.router)
app.include_router(reports.router)


@app.get("/health")
def health():
    return {"status": "ok"}
