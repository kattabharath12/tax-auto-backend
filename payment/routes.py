from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from uuid import uuid4
from datetime import datetime
from database import SessionLocal
from models import Payment
from auth.routes import get_current_user

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class PaymentRequest(BaseModel):
    amount: float
    payment_method: str

@router.post("/charge")
def make_payment(
    req: PaymentRequest,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    payment_id = str(uuid4())
    payment = Payment(
        id=payment_id,
        user_email=current_user.email,
        amount=req.amount,
        status="success",
        payment_method=req.payment_method,
        created_at=datetime.utcnow()
    )
    db.add(payment)
    db.commit()
    db.refresh(payment)
    return {"id": payment.id, "status": payment.status, "message": "Payment successful"}

@router.get("/")
def list_payments(current_user = Depends(get_current_user), db: Session = Depends(get_db)):
    payments = db.query(Payment).filter(Payment.user_email == current_user.email).all()
    return {
        "payments": [
            {
                "id": p.id,
                "amount": p.amount,
                "status": p.status,
                "payment_method": p.payment_method,
                "created_at": p.created_at
            } for p in payments
        ]
    }
