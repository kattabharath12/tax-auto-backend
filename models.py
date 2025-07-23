from sqlalchemy import Column, Integer, String, DateTime, Text, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    name = Column(String)
    ssn = Column(String)
    dob = Column(String)
    address = Column(Text)
    state = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, index=True)
    user_email = Column(String, index=True)
    filename = Column(String)
    file_path = Column(String)
    content_type = Column(String)
    document_type = Column(String)  # W-2, 1099-NEC, W-9, etc.
    extracted_data = Column(Text)  # JSON string
    uploaded_at = Column(DateTime, default=datetime.utcnow)

class TaxSubmission(Base):
    __tablename__ = "tax_submissions"
    id = Column(String, primary_key=True, index=True)
    user_email = Column(String, index=True)
    form_data = Column(Text)  # JSON string
    status = Column(String, default="pending")
    submitted_at = Column(DateTime, default=datetime.utcnow)
    tax_owed = Column(Float, default=0.0)
    refund_amount = Column(Float, default=0.0)

class Payment(Base):
    __tablename__ = "payments"
    id = Column(String, primary_key=True, index=True)
    user_email = Column(String, index=True)
    submission_id = Column(String)
    amount = Column(Float)
    status = Column(String, default="pending")
    payment_method = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class W9Form(Base):
    __tablename__ = "w9_forms"
    id = Column(String, primary_key=True, index=True)
    user_email = Column(String, index=True)
    document_id = Column(String)  # Reference to Document table
    name = Column(String)
    business_name = Column(String)
    tax_classification = Column(String)
    address = Column(Text)
    taxpayer_id = Column(String)
    ein = Column(String)
    ssn = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
