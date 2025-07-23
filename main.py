from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth import routes as auth_routes
from file_service import routes as file_routes
from tax_engine import routes as tax_routes
from submission import routes as submission_routes
from payment import routes as payment_routes
from admin import routes as admin_routes
# create_tables.py
from database import engine
from models import Base

Base.metadata.create_all(bind=engine)
print("Tables created!")

app = FastAPI(
    title="Tax Auto-Fill API",
    description="API for tax document upload, extraction, and filing (W-2, 1099, W-9, etc.)",
    version="1.0.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, set this to your frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth_routes.router, prefix="/api/auth", tags=["auth"])
app.include_router(file_routes.router, prefix="/api/files", tags=["files"])
app.include_router(tax_routes.router, prefix="/api/tax", tags=["tax"])
app.include_router(submission_routes.router, prefix="/api/submit", tags=["submission"])
app.include_router(payment_routes.router, prefix="/api/payments", tags=["payments"])
app.include_router(admin_routes.router, prefix="/api/admin", tags=["admin"])
