import os
import json
import re
import random
import platform
from typing import Dict, Any, Optional, List, Tuple

try:
    import pytesseract
    from PIL import Image
    import PyPDF2
    from pdf2image import convert_from_path
    import cv2
    import numpy as np
    
    # FORCE OCR to be available - override any detection issues
    OCR_AVAILABLE = True
    
    # Set Tesseract path based on platform
    if platform.system() == "Windows":
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    print("✅ OCR libraries loaded successfully - FORCED MODE")
    
except ImportError as e:
    print(f"⚠️  OCR libraries not available, using mock data: {e}")
    OCR_AVAILABLE = False

class DocumentProcessor:
    def __init__(self):
        # Generic W-2 box patterns that work for ANY W-2 document
        self.w2_box_patterns = {
            'wages': [
                # Box 1: Wages, tips, other compensation - look for number after the description
                r'(?:^|\n|\s)1\s+wages[,\s]*tips[,\s]*other compensation\s*[\n\s]*(\d{1,8}(?:\.\d{2})?)',
                r'(?:^|\n)1\s+wages.*?[\n\s]+(\d{1,8}(?:\.\d{2})?)',
                r'wages[,\s]*tips[,\s]*other compensation[\s\n]*(\d{1,8}(?:\.\d{2})?)',
                # Look for number in the wages box area (after "1" and wage-related text)
                r'1\s+(?:wages|salary).*?(\d{4,8}(?:\.\d{2})?)',
                # Generic pattern for first major number after wages mention
                r'(?:wages|salary|gross pay).*?(\d{4,8}(?:\.\d{2})?)',
                # Pattern for standalone large number (likely wages)
                r'(?:^|\n)\s*(\d{4,8}(?:\.\d{2})?)\s*(?:\n|$)',
            ],
            'federal_withholding': [
                # Box 2: Federal income tax withheld
                r'(?:^|\n|\s)2\s+federal income tax withheld\s*[\n\s]*(\d{1,6}(?:\.\d{2})?)',
                r'(?:^|\n)2\s+federal.*?[\n\s]+(\d{1,6}(?:\.\d{2})?)',
                r'federal income tax withheld[\s\n]*(\d{1,6}(?:\.\d{2})?)',
                r'2\s+federal.*?(\d{1,6}(?:\.\d{2})?)',
                # Look for smaller numbers associated with federal tax
                r'(?:federal tax|fed.*withh).*?(\d{1,6}(?:\.\d{2})?)',
            ],
            'social_security_wages': [
                # Box 3: Social security wages
                r'(?:^|\n|\s)3\s+social security wages\s*[\n\s]*(\d{1,8}(?:\.\d{2})?)',
                r'(?:^|\n)3\s+social.*?[\n\s]+(\d{1,8}(?:\.\d{2})?)',
                r'social security wages[\s\n]*(\d{1,8}(?:\.\d{2})?)',
                r'3\s+social.*?(\d{1,8}(?:\.\d{2})?)',
            ],
            'social_security_withholding': [
                # Box 4: Social security tax withheld
                r'(?:^|\n|\s)4\s+social security tax withheld\s*[\n\s]*(\d{1,6}(?:\.\d{2})?)',
                r'(?:^|\n)4\s+social.*tax.*?[\n\s]+(\d{1,6}(?:\.\d{2})?)',
                r'social security tax withheld[\s\n]*(\d{1,6}(?:\.\d{2})?)',
                r'4\s+social.*tax.*?(\d{1,6}(?:\.\d{2})?)',
            ],
            'medicare_wages': [
                # Box 5: Medicare wages and tips
                r'(?:^|\n|\s)5\s+medicare wages and tips\s*[\n\s]*(\d{1,8}(?:\.\d{2})?)',
                r'(?:^|\n)5\s+medicare.*?[\n\s]+(\d{1,8}(?:\.\d{2})?)',
                r'medicare wages and tips[\s\n]*(\d{1,8}(?:\.\d{2})?)',
                r'5\s+medicare.*?(\d{1,8}(?:\.\d{2})?)',
            ],
            'medicare_withholding': [
                # Box 6: Medicare tax withheld
                r'(?:^|\n|\s)6\s+medicare tax withheld\s*[\n\s]*(\d{1,6}(?:\.\d{2})?)',
                r'(?:^|\n)6\s+medicare.*tax.*?[\n\s]+(\d{1,6}(?:\.\d{2})?)',
                r'medicare tax withheld[\s\n]*(\d{1,6}(?:\.\d{2})?)',
                r'6\s+medicare.*tax.*?(\d{1,6}(?:\.\d{2})?)',
            ],
            'state_withholding': [
                # Box 17: State income tax
                r'(?:^|\n|\s)17\s+state income tax\s*[\n\s]*(\d{1,6}(?:\.\d{2})?)',
                r'(?:^|\n)17\s+state.*?[\n\s]+(\d{1,6}(?:\.\d{2})?)',
                r'state income tax[\s\n]*(\d{1,6}(?:\.\d{2})?)',
                r'17\s+state.*?(\d{1,6}(?:\.\d{2})?)',
            ],
            'employer_name': [
                # Box c: Employer's name - look for text after employer designation
                r'c\s+employers?\s+name.*?[\n\s]+([A-Za-z][A-Za-z\s&.,\-]{2,50}?)(?:\n|\d|$)',
                r'employer\s+name.*?[\n\s]+([A-Za-z][A-Za-z\s&.,\-]{2,50}?)(?:\n|$)',
                # Look for company-like names in the document
                r'([A-Z][A-Za-z\s&]{3,30}(?:Inc|LLC|Corp|Company|Co\.|Ltd)\.?)',
                # Extract multi-word capitalized names
                r'([A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)*)',
            ],
            'employer_ein': [
                # Box b: Employer identification number - look for EIN format
                r'b\s+employer identification number.*?[\n\s]+([A-Z0-9\-]{9,12})',
                r'employer identification number.*?[\n\s]+([A-Z0-9\-]{9,12})',
                r'ein[:\s]*([A-Z0-9\-]{9,12})',
                # Standard EIN format: XX-XXXXXXX
                r'(\d{2}-\d{7})',
                # Alternative formats
                r'([A-Z]{2,4}\d{7,9})',
            ]
        }

    def extract_document_data(self, file_path: str, content_type: str) -> Dict[str, Any]:
        """Main method to extract data from uploaded documents"""
        
        print(f"🔍 DEBUG: OCR_AVAILABLE = {OCR_AVAILABLE}")
        
        if not OCR_AVAILABLE:
            print("Using mock data - OCR not available")
            return self._generate_mock_data(file_path)
            
        try:
            print(f"Processing document with GENERIC W-2 patterns: {os.path.basename(file_path)}")
            
            # Extract text from document
            extracted_text = self._extract_text_from_file(file_path, content_type)
            
            if not extracted_text or len(extracted_text.strip()) < 10:
                print("OCR extraction failed or insufficient text, using mock data")
                return self._generate_mock_data(file_path)
            
            print(f"Extracted text length: {len(extracted_text)} characters")
            
            # Clean and normalize text for better pattern matching
            normalized_text = self._normalize_text_for_w2(extracted_text)
            
            # Show sample of extracted text for debugging
            print("=== EXTRACTED TEXT SAMPLE ===")
            print(extracted_text[:500])
            print("=== END SAMPLE ===")
            
            # Determine document type
            doc_type = self._identify_document_type(normalized_text, file_path)
            print(f"Identified document type: {doc_type}")
            
            if doc_type == "W-2":
                return self._extract_w2_data_generic(normalized_text, extracted_text)
            else:
                return self._extract_generic_data(normalized_text)
                
        except Exception as e:
            print(f"OCR processing failed: {e}, falling back to mock data")
            import traceback
            traceback.print_exc()
            return self._generate_mock_data(file_path)

    def _extract_text_from_file(self, file_path: str, content_type: str) -> str:
        """Extract text using multiple methods for maximum coverage"""
        all_text = ""
        
        try:
            if content_type == "application/pdf":
                # Method 1: Direct PDF text extraction
                try:
                    with open(file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        pdf_text = ""
                        for page in pdf_reader.pages:
                            page_text = page.extract_text()
                            if page_text.strip():
                                pdf_text += page_text + "\n"
                        
                        if pdf_text.strip():
                            all_text += "=== PDF DIRECT EXTRACTION ===\n" + pdf_text + "\n"
                            print(f"PDF direct extraction: {len(pdf_text)} characters")
                except Exception as e:
                    print(f"PDF direct extraction failed: {e}")
                
                # Method 2: OCR on PDF images
                try:
                    images = convert_from_path(file_path, dpi=300, first_page=1, last_page=1)
                    if images:
                        # Try multiple OCR configurations
                        ocr_configs = [
                            '--psm 6 --oem 3',  # Uniform block of text
                            '--psm 4 --oem 3',  # Single column
                            '--psm 12 --oem 3', # Sparse text
                        ]
                        
                        best_ocr = ""
                        best_score = 0
                        
                        for config in ocr_configs:
                            try:
                                ocr_text = pytesseract.image_to_string(images[0], config=config)
                                score = self._score_text_quality(ocr_text)
                                
                                if score > best_score:
                                    best_ocr = ocr_text
                                    best_score = score
                                    print(f"Better OCR with {config}: score {score:.2f}")
                            except:
                                continue
                        
                        if best_ocr:
                            all_text += "=== OCR EXTRACTION ===\n" + best_ocr + "\n"
                            print(f"OCR extraction: {len(best_ocr)} characters")
                
                except Exception as e:
                    print(f"PDF OCR failed: {e}")
            
            elif content_type.startswith("image/"):
                # Image OCR
                try:
                    image = cv2.imread(file_path)
                    if image is not None:
                        # Preprocess image
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        # Enhance contrast
                        enhanced = cv2.equalizeHist(gray)
                        # Denoise
                        denoised = cv2.fastNlMeansDenoising(enhanced)
                        
                        # OCR
                        ocr_text = pytesseract.image_to_string(denoised, config='--psm 6 --oem 3')
                        all_text += "=== IMAGE OCR ===\n" + ocr_text + "\n"
                        print(f"Image OCR: {len(ocr_text)} characters")
                    else:
                        # Fallback to PIL
                        pil_image = Image.open(file_path)
                        ocr_text = pytesseract.image_to_string(pil_image)
                        all_text += "=== PIL OCR ===\n" + ocr_text + "\n"
                        print(f"PIL OCR: {len(ocr_text)} characters")
                
                except Exception as e:
                    print(f"Image OCR failed: {e}")
        
        except Exception as e:
            print(f"Text extraction error: {e}")
        
        return all_text

    def _score_text_quality(self, text: str) -> float:
        """Score the quality of extracted text"""
        if not text.strip():
            return 0.0
        
        score = 0.0
        
        # Check for W-2 indicators
        w2_terms = ['w-2', 'wage', 'tax', 'withheld', 'social security', 'medicare', 'federal', 'employer']
        found_terms = sum(1 for term in w2_terms if term in text.lower())
        score += (found_terms / len(w2_terms)) * 0.4
        
        # Check for numbers (tax forms have many numbers)
        digit_count = len(re.findall(r'\d', text))
        digit_ratio = digit_count / max(len(text), 1)
        score += min(digit_ratio * 5, 0.3)
        
        # Check for box numbers
        box_numbers = len(re.findall(r'\b(?:[1-9]|1[0-9]|20)\b', text))
        score += min(box_numbers * 0.02, 0.3)
        
        return min(score, 1.0)

    def _normalize_text_for_w2(self, text: str) -> str:
        """Normalize text for better W-2 pattern matching"""
        # Convert to lowercase for easier matching
        normalized = text.lower()
        
        # Clean up extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Ensure clear separation around numbers
        normalized = re.sub(r'(\d)([a-z])', r'\1 \2', normalized)
        normalized = re.sub(r'([a-z])(\d)', r'\1 \2', normalized)
        
        # Clean up common OCR artifacts
        normalized = re.sub(r'[^\w\s\-\.,\n]', ' ', normalized)
        
        # Ensure line breaks are preserved for box structure
        normalized = re.sub(r'\n+', '\n', normalized)
        
        return normalized

    def _extract_w2_data_generic(self, normalized_text: str, original_text: str) -> Dict[str, Any]:
        """Generic W-2 data extraction that works for any W-2"""
        data = {
            "document_type": "W-2",
            "confidence": 0.0,
            "extraction_method": "Generic Pattern Matching",
            "debug_info": [],
            "field_details": {}
        }
        
        texts_to_search = [normalized_text, original_text, original_text.lower()]
        successful_extractions = 0
        total_fields = len(self.w2_box_patterns)
        
        for field, patterns in self.w2_box_patterns.items():
            best_value = None
            best_confidence = 0
            attempts = []
            
            for text_version in texts_to_search:
                for i, pattern in enumerate(patterns):
                    try:
                        matches = re.findall(pattern, text_version, re.IGNORECASE | re.MULTILINE)
                        
                        for match in matches:
                            value = match.strip() if isinstance(match, str) else str(match)
                            
                            if field in ['wages', 'federal_withholding', 'social_security_wages', 
                                       'social_security_withholding', 'medicare_wages', 'medicare_withholding', 'state_withholding']:
                                # Numeric fields
                                cleaned_value = self._clean_currency_safe(value)
                                
                                if cleaned_value is not None and 0 <= cleaned_value <= 1000000:
                                    confidence = self._calculate_field_confidence(field, cleaned_value, text_version)
                                    attempts.append({
                                        'pattern': f"Pattern {i+1}",
                                        'raw_match': value,
                                        'cleaned_value': cleaned_value,
                                        'confidence': confidence
                                    })
                                    
                                    if confidence > best_confidence:
                                        best_value = cleaned_value
                                        best_confidence = confidence
                            else:
                                # Text fields
                                if 2 <= len(value) <= 50 and not value.isdigit():
                                    confidence = 0.6 if len(value) > 5 else 0.4
                                    attempts.append({
                                        'pattern': f"Pattern {i+1}",
                                        'raw_match': value,
                                        'cleaned_value': value,
                                        'confidence': confidence
                                    })
                                    
                                    if confidence > best_confidence:
                                        best_value = value
                                        best_confidence = confidence
                    
                    except Exception as e:
                        attempts.append({
                            'pattern': f"Pattern {i+1}",
                            'error': str(e)
                        })
            
            # Store results
            if best_value is not None and best_confidence > 0.3:  # Minimum confidence threshold
                data[field] = best_value
                data['field_details'][field] = {
                    'value': best_value,
                    'confidence': best_confidence,
                    'attempts': attempts
                }
                successful_extractions += 1
                data["debug_info"].append(f"✅ {field}: {best_value} (confidence: {best_confidence:.2f})")
            else:
                data["debug_info"].append(f"❌ {field}: No reliable extraction")
                data['field_details'][field] = {
                    'value': None,
                    'confidence': 0,
                    'attempts': attempts
                }
        
        # Set defaults for missing critical fields
        data.setdefault('wages', 0.0)
        data.setdefault('federal_withholding', 0.0)
        data.setdefault('state_withholding', 0.0)
        data.setdefault('employer_name', 'Not found')
        
        # Calculate overall confidence
        data['confidence'] = successful_extractions / total_fields if total_fields > 0 else 0
        
        print(f"\n=== W-2 EXTRACTION SUMMARY ===")
        print(f"Successful extractions: {successful_extractions}/{total_fields}")
        print(f"Overall confidence: {data['confidence']:.2f}")
        
        # Show key extracted values
        key_fields = ['wages', 'federal_withholding', 'social_security_wages', 'medicare_wages']
        for field in key_fields:
            if field in data and isinstance(data[field], (int, float)):
                print(f"{field}: ${data[field]:,.2f}")
        
        return data

    def _clean_currency_safe(self, value: str) -> Optional[float]:
        """Safely clean currency values with strict validation"""
        if not value or not isinstance(value, str):
            return None
        
        # Remove common formatting
        cleaned = re.sub(r'[$,\s]', '', value.strip())
        
        # Remove any non-numeric characters except decimal point
        cleaned = re.sub(r'[^\d.]', '', cleaned)
        
        # Handle multiple decimal points
        if cleaned.count('.') > 1:
            parts = cleaned.split('.')
            cleaned = parts[0] + '.' + ''.join(parts[1:])
        
        # Must have at least one digit
        if not re.search(r'\d', cleaned):
            return None
        
        try:
            result = float(cleaned)
            # Reasonable bounds for W-2 values
            if 0 <= result <= 10000000:
                return result
        except (ValueError, OverflowError):
            pass
        
        return None

    def _calculate_field_confidence(self, field: str, value: float, context: str) -> float:
        """Calculate confidence based on field type and value reasonableness"""
        confidence = 0.4  # Base confidence
        
        # Field-specific validation
        if field == 'wages':
            if 1000 <= value <= 500000:
                confidence += 0.4
            elif 10000 <= value <= 200000:
                confidence += 0.5  # Most common range
        elif field in ['federal_withholding', 'state_withholding']:
            if 0 <= value <= 100000:
                confidence += 0.3
            if value > 0:  # Non-zero withholding is expected
                confidence += 0.2
        elif field in ['social_security_wages', 'medicare_wages']:
            if 100 <= value <= 200000:
                confidence += 0.4
        elif field in ['social_security_withholding', 'medicare_withholding']:
            if 0 <= value <= 20000:
                confidence += 0.3
        
        # Context bonus (if found near relevant keywords)
        field_keywords = {
            'wages': ['wage', 'salary', 'gross', 'compensation'],
            'federal_withholding': ['federal', 'tax', 'withheld'],
            'social_security_wages': ['social security', 'ss wages'],
            'medicare_wages': ['medicare', 'med wages']
        }
        
        if field in field_keywords:
            for keyword in field_keywords[field]:
                if keyword in context.lower():
                    confidence += 0.1
                    break
        
        return min(confidence, 1.0)

    def _identify_document_type(self, text: str, filename: str) -> str:
        """Identify document type"""
        text_lower = text.lower()
        filename_lower = os.path.basename(filename).lower()
        
        # Check filename
        if any(keyword in filename_lower for keyword in ['w2', 'w-2']):
            return "W-2"
        
        # Check content
        w2_score = 0
        w2_indicators = ['w-2', 'wage and tax statement', 'employer identification', 
                        'federal income tax withheld', 'social security wages', 'medicare']
        
        for indicator in w2_indicators:
            if indicator in text_lower:
                w2_score += 1
        
        # Check for numbered boxes typical of W-2
        box_matches = len(re.findall(r'\b[1-9]\s+(?:wages|federal|social|medicare)', text_lower))
        w2_score += box_matches
        
        if w2_score >= 2:
            return "W-2"
        
        return "Unknown"

    def _extract_generic_data(self, text: str) -> Dict[str, Any]:
        """Extract data from unknown document types"""
        return {
            "document_type": "Unknown",
            "extracted_text": text[:500] + "..." if len(text) > 500 else text,
            "confidence": 0.3,
            "extraction_method": "Generic",
            "message": "Document type not recognized. Please verify the extracted information."
        }

    def _generate_mock_data(self, file_path: str) -> Dict[str, Any]:
        """Generate mock data when OCR fails"""
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
