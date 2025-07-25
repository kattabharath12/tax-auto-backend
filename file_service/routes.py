from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import uuid4
import os
import json
from database import SessionLocal
from models import Document
from auth.routes import get_current_user
from .ocr_mock import extract_document_data

router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    file_id = str(uuid4())
    filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    # Extract data (mock)
    extracted_data = extract_document_data(file_path, file.content_type)
    doc = Document(
        id=file_id,
        user_email=current_user.email,
        filename=file.filename,
        file_path=file_path,
        content_type=file.content_type,
        document_type=extracted_data.get("document_type", "Unknown"),
        extracted_data=json.dumps(extracted_data)
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return {
        "id": doc.id,
        "filename": doc.filename,
        "extracted_data": extracted_data,
        "uploaded_at": doc.uploaded_at
    }

@router.get("/")
def list_files(current_user = Depends(get_current_user), db: Session = Depends(get_db)):
    docs = db.query(Document).filter(Document.user_email == current_user.email).all()
    return {
        "documents": [
            {
                "id": d.id,
                "filename": d.filename,
                "extracted_data": json.loads(d.extracted_data),
                "uploaded_at": d.uploaded_at
            } for d in docs
        ]
    }
# ADD THIS TO file_service/routes.py
@router.get("/user-documents")
async def get_user_documents(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all documents for the current user"""
    try:
        docs = db.query(Document).filter(Document.user_email == current_user.email).all()
        
        documents = []
        for doc in docs:
            doc_data = {
                "id": doc.id,
                "filename": doc.filename,
                "document_type": doc.document_type,
                "upload_date": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
                "extracted_data": json.loads(doc.extracted_data) if doc.extracted_data else None
            }
            documents.append(doc_data)
        
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch documents: {str(e)}")

# ADD THIS TOO
@router.get("/download/{document_id}")
async def download_file(
    document_id: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Download a specific document"""
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.user_email == current_user.email
    ).first()
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"download_url": f"/uploads/{doc.filename}", "filename": doc.filename}
