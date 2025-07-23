import random
from typing import Dict, Any

def extract_document_data(file_path: str, content_type: str) -> Dict[str, Any]:
    filename = file_path.lower()
    if "w2" in filename or "w-2" in filename:
        return extract_w2_data()
    elif "1099" in filename:
        return extract_1099_data()
    elif "w9" in filename or "w-9" in filename:
        return extract_w9_data()
    else:
        return extract_generic_tax_document()

def extract_w2_data() -> Dict[str, Any]:
    return {
        "document_type": "W-2",
        "employer_name": "Demo Corp Inc",
        "employer_ein": "12-3456789",
        "employee_ssn": "***-**-1234",
        "wages": round(random.uniform(40000, 120000), 2),
        "federal_withholding": round(random.uniform(5000, 20000), 2),
        "social_security_wages": round(random.uniform(40000, 120000), 2),
        "social_security_withholding": round(random.uniform(2000, 8000), 2),
        "medicare_wages": round(random.uniform(40000, 120000), 2),
        "medicare_withholding": round(random.uniform(600, 1800), 2),
        "state_wages": round(random.uniform(40000, 120000), 2),
        "state_withholding": round(random.uniform(2000, 8000), 2),
        "confidence": 0.95
    }

def extract_1099_data() -> Dict[str, Any]:
    return {
        "document_type": "1099-NEC",
        "payer_name": "Freelance Client LLC",
        "payer_tin": "98-7654321",
        "recipient_ssn": "***-**-1234",
        "nonemployee_compensation": round(random.uniform(5000, 50000), 2),
        "federal_withholding": round(random.uniform(0, 5000), 2),
        "state_withholding": round(random.uniform(0, 2000), 2),
        "confidence": 0.92
    }

def extract_w9_data() -> Dict[str, Any]:
    business_types = ["Individual/sole proprietor", "C Corporation", "S Corporation", "Partnership", "LLC"]
    return {
        "document_type": "W-9",
        "name": "John Doe Business Services",
        "business_name": "Doe Consulting LLC",
        "federal_tax_classification": random.choice(business_types),
        "address": "123 Business St, Suite 100",
        "city": "Business City",
        "state": "CA",
        "zip_code": "90210",
        "taxpayer_id": "12-3456789",
        "ssn": "***-**-1234",
        "ein": "12-3456789",
        "account_numbers": "1234567890",
        "requester_name": "Client Company Inc",
        "requester_address": "456 Client Ave, Client City, CA 90211",
        "confidence": 0.93,
        "extracted_fields": [
            "name", "business_name", "tax_classification", 
            "address", "taxpayer_id", "account_numbers"
        ]
    }

def extract_generic_tax_document() -> Dict[str, Any]:
    return {
        "document_type": "Unknown",
        "extracted_text": "Sample extracted text from document",
        "confidence": 0.75,
        "fields_detected": ["income", "withholding", "employer_info"],
        "message": "Document uploaded successfully. Please review and edit the extracted information."
    }
