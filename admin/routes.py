from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import SessionLocal
from models import TaxSubmission, Payment
from auth.routes import get_current_user

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def is_admin(user):
    # For demo, treat the first registered user as admin
    return user.email.endswith("@admin.com") or user.email == "admin@example.com"

@router.get("/submissions")
def get_all_submissions(current_user = Depends(get_current_user), db: Session = Depends(get_db)):
    if not is_admin(current_user):
        return {"error": "Admin access required"}
    subs = db.query(TaxSubmission).all()
    return {
        "submissions": [
            {
                "id": s.id,
                "user_email": s.user_email,
                "status": s.status,
                "submitted_at": s.submitted_at,
                "filing_type": s.form_data,
                "tax_owed": s.tax_owed,
                "refund_amount": s.refund_amount
            } for s in subs
        ]
    }

@router.get("/payments")
def get_all_payments(current_user = Depends(get_current_user), db: Session = Depends(get_db)):
    if not is_admin(current_user):
        return {"error": "Admin access required"}
    pays = db.query(Payment).all()
    return {
        "payments": [
            {
                "id": p.id,
                "user_email": p.user_email,
                "amount": p.amount,
                "status": p.status,
                "payment_method": p.payment_method,
                "created_at": p.created_at
            } for p in pays
        ]
    }

@router.get("/stats")
def get_stats(current_user = Depends(get_current_user), db: Session = Depends(get_db)):
    if not is_admin(current_user):
        return {"error": "Admin access required"}
    total_submissions = db.query(TaxSubmission).count()
    total_payments = db.query(Payment).count()
    return {
        "total_submissions": total_submissions,
        "total_payments": total_payments,
        "submission_stats": {},
        "payment_stats": {}
    }
