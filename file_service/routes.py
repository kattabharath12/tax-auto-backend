from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from uuid import uuid4
import os
import json
from database import SessionLocal
from models import Document
from auth.routes import get_current_user
# Replace the mock import with real processor
from file_service.real_ocr_processor import extract_document_data

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
    """Upload and process document with real OCR"""
    try:
        # Validate file type
        allowed_types = ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed types: {', '.join(allowed_types)}"
            )
        
        # Generate unique file ID and save file
        file_id = str(uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # Save uploaded file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Extract data using real OCR
        print(f"Processing document: {file.filename}")
        extracted_data = extract_document_data(file_path, file.content_type)
        
        # Save document info to database
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
        
        print(f"Document processed successfully: {extracted_data.get('document_type', 'Unknown')}")
        
        return {
            "id": doc.id,
            "filename": doc.filename,
            "document_type": extracted_data.get("document_type"),
            "extracted_data": extracted_data,
            "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
            "confidence": extracted_data.get("confidence", 0.0)
        }
        
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/user-documents")
async def get_user_documents(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all documents for the current user with extracted data"""
    try:
        docs = db.query(Document).filter(Document.user_email == current_user.email).all()
        
        documents = []
        for doc in docs:
            # Parse extracted data
            extracted_data = None
            if doc.extracted_data:
                try:
                    extracted_data = json.loads(doc.extracted_data)
                except json.JSONDecodeError:
                    extracted_data = {"error": "Failed to parse extracted data"}
            
            doc_data = {
                "id": doc.id,
                "filename": doc.filename,
                "document_type": doc.document_type,
                "upload_date": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
                "extracted_data": extracted_data,
                "confidence": extracted_data.get("confidence", 0.0) if extracted_data else 0.0
            }
            documents.append(doc_data)
        
        return documents
    except Exception as e:
        print(f"Error fetching documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch documents: {str(e)}")

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
    
    if not os.path.exists(doc.file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    return FileResponse(
        path=doc.file_path,
        filename=doc.filename,
        media_type=doc.content_type
    )

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a document"""
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.user_email == current_user.email
    ).first()
    
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete file from disk
    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)
    
    # Delete from database
    db.delete(doc)
    db.commit()
    
    return {"message": "Document deleted successfully"}

@router.get("/")
def list_files(current_user = Depends(get_current_user), db: Session = Depends(get_db)):
    """Legacy endpoint - redirects to user-documents"""
    return get_user_documents(current_user, db)
