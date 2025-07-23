from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, Any
from auth.routes import get_current_user
from .calculator import TaxCalculator

router = APIRouter()

class TaxCalculationRequest(BaseModel):
    form_data: Dict[str, Any]
    filing_status: str = "single"
    state: str = "CA"

@router.post("/calculate")
async def calculate_taxes(
    request: TaxCalculationRequest,
    current_user = Depends(get_current_user)
):
    calculator = TaxCalculator()
    result = calculator.calculate(
        form_data=request.form_data,
        filing_status=request.filing_status,
        state=request.state
    )
    return result

@router.get("/forms/{form_type}")
async def get_form_template(form_type: str):
    templates = {
        "1040": {
            "name": "Form 1040",
            "fields": [
                {"name": "wages", "label": "Wages (W-2)", "type": "number"},
                {"name": "interest", "label": "Interest Income", "type": "number"},
                {"name": "dividends", "label": "Dividends", "type": "number"},
                {"name": "business_income", "label": "Business Income", "type": "number"},
                {"name": "federal_withholding", "label": "Federal Tax Withheld", "type": "number"}
            ]
        },
        "schedule_a": {
            "name": "Schedule A - Itemized Deductions",
            "fields": [
                {"name": "medical_expenses", "label": "Medical Expenses", "type": "number"},
                {"name": "state_local_taxes", "label": "State/Local Taxes", "type": "number"},
                {"name": "mortgage_interest", "label": "Mortgage Interest", "type": "number"},
                {"name": "charitable_contributions", "label": "Charitable Contributions", "type": "number"}
            ]
        },
        "schedule_c": {
            "name": "Schedule C - Business Income",
            "fields": [
                {"name": "gross_receipts", "label": "Gross Receipts", "type": "number"},
                {"name": "business_expenses", "label": "Business Expenses", "type": "number"},
                {"name": "home_office", "label": "Home Office Deduction", "type": "number"}
            ]
        },
        "w9": {
            "name": "Form W-9 - Request for Taxpayer Identification Number",
            "fields": [
                {"name": "name", "label": "Name", "type": "text"},
                {"name": "business_name", "label": "Business Name", "type": "text"},
                {"name": "tax_classification", "label": "Federal Tax Classification", "type": "text"},
                {"name": "address", "label": "Address", "type": "text"},
                {"name": "city", "label": "City", "type": "text"},
                {"name": "state", "label": "State", "type": "text"},
                {"name": "zip_code", "label": "ZIP Code", "type": "text"},
                {"name": "taxpayer_id", "label": "Taxpayer Identification Number", "type": "text"},
                {"name": "ssn", "label": "Social Security Number", "type": "text"},
                {"name": "ein", "label": "Employer Identification Number", "type": "text"},
                {"name": "account_numbers", "label": "Account Number(s)", "type": "text"},
                {"name": "requester_name", "label": "Requester's Name", "type": "text"},
                {"name": "requester_address", "label": "Requester's Address", "type": "text"}
            ]
        }
    }
    if form_type not in templates:
        return {"error": "Form type not found"}
    return templates[form_type]
