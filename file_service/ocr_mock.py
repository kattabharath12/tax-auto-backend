import os
import json
import re
import random
import platform
from typing import Dict, Any, Optional

# Try to import OCR libraries, fallback to mock if not available
try:
    import pytesseract
    from PIL import Image
    import PyPDF2
    from pdf2image import convert_from_path
    import cv2
    import numpy as np
    OCR_AVAILABLE = True
    
    # Set Tesseract path based on platform
    if platform.system() == "Windows":
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # On Linux (Railway), tesseract should be in PATH automatically
    
    print("✅ OCR libraries loaded successfully")
    
except ImportError as e:
    print(f"⚠️  OCR libraries not available, using mock data: {e}")
    OCR_AVAILABLE = False

class DocumentProcessor:
    def __init__(self):
        self.w2_patterns = {
            'employer_name': r'(?:employer|company|corp|inc|llc)[:\s]*([A-Za-z\s&.,\-]+?)(?:\n|$)',
            'employer_ein': r'(?:ein|employer id|identification)[:\s]*(\d{2}-\d{7})',
            'wages': r'(?:wages|salary|gross)[:\s]*\$?([\d,]+\.?\d*)',
            'federal_withholding': r'(?:federal|fed).*?(?:withh|tax)[:\s]*\$?([\d,]+\.?\d*)',
            'social_security_wages': r'(?:social security|ss).*?(?:wages)[:\s]*\$?([\d,]+\.?\d*)',
            'medicare_wages': r'(?:medicare).*?(?:wages)[:\s]*\$?([\d,]+\.?\d*)',
            'state_withholding': r'(?:state).*?(?:withh|tax)[:\s]*\$?([\d,]+\.?\d*)',
        }
        
        self.form_1099_patterns = {
            'payer_name': r'(?:payer|company|from)[:\s]*([A-Za-z\s&.,\-]+?)(?:\n|$)',
            'payer_tin': r'(?:payer|tin|id)[:\s]*(\d{2}-\d{7})',
            'nonemployee_compensation': r'(?:nonemployee|compensation|1099)[:\s]*\$?([\d,]+\.?\d*)',
            'federal_withholding': r'(?:federal|backup).*?(?:withh)[:\s]*\$?([\d,]+\.?\d*)',
        }
        
        self.w9_patterns = {
            'name': r'(?:name)[:\s]*([A-Za-z\s\-\']+?)(?:\n|business)',
            'business_name': r'(?:business|company)[:\s]*([A-Za-z\s&.,\-]+?)(?:\n|$)',
            'address': r'(?:address)[:\s]*([A-Za-z0-9\s,.\-#]+?)(?:\n|city)',
            'city': r'(?:city)[:\s]*([A-Za-z\s\-]+?)(?:\n|state)',
            'state': r'(?:state)[:\s]*([A-Z]{2})',
            'zip_code': r'(?:zip|postal)[:\s]*(\d{5}(?:-\d{4})?)',
            'taxpayer_id': r'(?:ssn|ein|tin)[:\s]*(\d{3}-\d{2}-\d{4}|\d{2}-\d{7})',
        }

    def extract_document_data(self, file_path: str, content_type: str) -> Dict[str, Any]:
        """Main method to extract data from uploaded documents"""
        if not OCR_AVAILABLE:
            print("Using mock data - OCR not available")
            return self._generate_mock_data(file_path)
        
        try:
            print(f"Processing document with OCR: {os.path.basename(file_path)}")
            
            # Extract text from document
            extracted_text = self._extract_text_from_file(file_path, content_type)
            
            if not extracted_text or len(extracted_text.strip()) < 10:
                print("OCR extraction failed or insufficient text, using mock data")
                return self._generate_mock_data(file_path)
            
            # Determine document type and extract relevant data
            doc_type = self._identify_document_type(extracted_text, file_path)
            print(f"Identified document type: {doc_type}")
            
            if doc_type == "W-2":
                return self._extract_w2_data(extracted_text)
            elif doc_type == "1099-NEC":
                return self._extract_1099_data(extracted_text)
            elif doc_type == "W-9":
                return self._extract_w9_data(extracted_text)
            else:
                return self._extract_generic_data(extracted_text)
                
        except Exception as e:
            print(f"OCR processing failed: {e}, falling back to mock data")
            return self._generate_mock_data(file_path)

    def _extract_text_from_file(self, file_path: str, content_type: str) -> str:
        """Extract text from various file types"""
        try:
            if content_type == "application/pdf":
                return self._extract_text_from_pdf(file_path)
            elif content_type.startswith("image/"):
                return self._extract_text_from_image(file_path)
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
        except Exception as e:
            print(f"Text extraction error: {e}")
            return ""

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using PyPDF2 and OCR fallback"""
        text = ""
        
        try:
            # First try direct text extraction
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += page_text + "\n"
            
            # If no text found, use OCR on PDF images
            if not text.strip():
                print("Direct PDF text extraction failed, trying OCR...")
                images = convert_from_path(file_path)
                for image in images:
                    text += pytesseract.image_to_string(image) + "\n"
                    
        except Exception as e:
            print(f"PDF processing error: {e}")
        
        return text

    def _extract_text_from_image(self, file_path: str) -> str:
        """Extract text from image using OCR"""
        try:
            # Preprocess image for better OCR
            image = cv2.imread(file_path)
            if image is None:
                # Fallback to PIL
                pil_image = Image.open(file_path)
                return pytesseract.image_to_string(pil_image)
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Use pytesseract for OCR
            text = pytesseract.image_to_string(gray, config='--psm 6')
            return text
            
        except Exception as e:
            print(f"Image OCR error: {e}")
            return ""

    def _identify_document_type(self, text: str, filename: str) -> str:
        """Identify document type based on content and filename"""
        text_lower = text.lower()
        filename_lower = os.path.basename(filename).lower()
        
        # Check filename first for hints
        if any(keyword in filename_lower for keyword in ['w2', 'w-2']):
            return "W-2"
        elif any(keyword in filename_lower for keyword in ['1099', '1099-nec']):
            return "1099-NEC"
        elif any(keyword in filename_lower for keyword in ['w9', 'w-9']):
            return "W-9"
        
        # Check document content
        if any(keyword in text_lower for keyword in ['wage and tax statement', 'w-2', 'employer identification']):
            return "W-2"
        elif any(keyword in text_lower for keyword in ['1099-nec', 'nonemployee compensation']):
            return "1099-NEC"
        elif any(keyword in text_lower for keyword in ['w-9', 'request for taxpayer', 'taxpayer identification']):
            return "W-9"
        
        return "Unknown"

    def _extract_w2_data(self, text: str) -> Dict[str, Any]:
        """Extract W-2 specific data"""
        data = {
            "document_type": "W-2",
            "confidence": 0.0,
            "extraction_method": "OCR"
        }
        
        matches_found = 0
        
        for field, pattern in self.w2_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1).strip()
                
                if field in ['wages', 'federal_withholding', 'social_security_wages', 'medicare_wages', 'state_withholding']:
                    value = self._clean_currency(value)
                    data[field] = value
                else:
                    data[field] = value
                
                matches_found += 1
        
        data['confidence'] = min(0.95, matches_found / len(self.w2_patterns) * 1.2)
        
        # Set default values for missing fields
        data.setdefault('employer_name', 'Not found')
        data.setdefault('wages', 0.0)
        data.setdefault('federal_withholding', 0.0)
        data.setdefault('state_withholding', 0.0)
        
        return data

    def _extract_1099_data(self, text: str) -> Dict[str, Any]:
        """Extract 1099-NEC specific data"""
        data = {
            "document_type": "1099-NEC",
            "confidence": 0.0,
            "extraction_method": "OCR"
        }
        
        matches_found = 0
        
        for field, pattern in self.form_1099_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1).strip()
                
                if field in ['nonemployee_compensation', 'federal_withholding']:
                    value = self._clean_currency(value)
                    data[field] = value
                else:
                    data[field] = value
                
                matches_found += 1
        
        data['confidence'] = min(0.95, matches_found / len(self.form_1099_patterns) * 1.2)
        
        data.setdefault('payer_name', 'Not found')
        data.setdefault('nonemployee_compensation', 0.0)
        data.setdefault('federal_withholding', 0.0)
        
        return data

    def _extract_w9_data(self, text: str) -> Dict[str, Any]:
        """Extract W-9 specific data"""
        data = {
            "document_type": "W-9",
            "confidence": 0.0,
            "extraction_method": "OCR"
        }
        
        matches_found = 0
        
        for field, pattern in self.w9_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1).strip()
                data[field] = value
                matches_found += 1
        
        data['confidence'] = min(0.95, matches_found / len(self.w9_patterns) * 1.2)
        
        data.setdefault('name', 'Not found')
        data.setdefault('address', 'Not found')
        
        return data

    def _extract_generic_data(self, text: str) -> Dict[str, Any]:
        """Extract data from unknown document types"""
        return {
            "document_type": "Unknown",
            "extracted_text": text[:500] + "..." if len(text) > 500 else text,
            "confidence": 0.3,
            "extraction_method": "OCR",
            "message": "Document type not recognized. Please verify the extracted information."
        }

    def _clean_currency(self, value: str) -> float:
        """Clean and convert currency strings to float"""
        if not value:
            return 0.0
        
        cleaned = re.sub(r'[$,\s]', '', value)
        
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    def _generate_mock_data(self, file_path: str) -> Dict[str, Any]:
        """Fallback mock data generator"""
        filename = os.path.basename(file_path).lower()
        
        if "w2" in filename or "w-2" in filename:
            return {
                "document_type": "W-2",
                "employer_name": "Demo Corp Inc",
                "wages": round(random.uniform(40000, 120000), 2),
                "federal_withholding": round(random.uniform(5000, 20000), 2),
                "state_withholding": round(random.uniform(2000, 8000), 2),
                "confidence": 0.85,
                "extraction_method": "Mock",
                "note": "Mock data - OCR not available"
            }
        elif "1099" in filename:
            return {
                "document_type": "1099-NEC",
                "payer_name": "Client Company LLC",
                "nonemployee_compensation": round(random.uniform(5000, 50000), 2),
                "federal_withholding": round(random.uniform(0, 5000), 2),
                "confidence": 0.85,
                "extraction_method": "Mock",
                "note": "Mock data - OCR not available"
            }
        else:
            return {
                "document_type": "Unknown",
                "confidence": 0.5,
                "extraction_method": "Mock",
                "note": "Mock data - OCR not available"
            }

# Create global processor instance
processor = DocumentProcessor()

def extract_document_data(file_path: str, content_type: str) -> Dict[str, Any]:
    """Main function to extract real data from documents"""
    return processor.extract_document_data(file_path, content_type)
