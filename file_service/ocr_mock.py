# Replace the _extract_text_from_file method in your ocr_mock.py with this fixed version:

def _extract_text_from_file(self, file_path: str, content_type: str) -> str:
    """Extract text using multiple methods for maximum coverage - FIXED for image PDFs"""
    all_text = ""
    
    try:
        if content_type == "application/pdf":
            print(f"Processing PDF: {file_path}")
            
            # Method 1: Direct PDF text extraction
            direct_text = ""
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text.strip():
                            direct_text += page_text + "\n"
                
                if direct_text.strip():
                    print(f"✅ PDF direct extraction successful: {len(direct_text)} characters")
                    all_text += "=== PDF DIRECT EXTRACTION ===\n" + direct_text + "\n"
                else:
                    print("⚠️  PDF direct extraction returned empty - trying OCR")
                    
            except Exception as e:
                print(f"❌ PDF direct extraction failed: {e}")
            
            # Method 2: OCR on PDF images (ALWAYS try this for tax documents)
            print("🔄 Attempting OCR on PDF images...")
            try:
                # Convert PDF to images with high DPI for better OCR
                images = convert_from_path(file_path, dpi=400, first_page=1, last_page=1)
                print(f"📄 Converted PDF to {len(images)} image(s)")
                
                if images:
                    image = images[0]  # Process first page
                    print(f"🖼️  Image size: {image.size}")
                    
                    # Try multiple OCR configurations for tax forms
                    ocr_configs = [
                        '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,\n-',
                        '--psm 4 --oem 3',
                        '--psm 6 --oem 3',
                        '--psm 12 --oem 3',
                    ]
                    
                    best_ocr_text = ""
                    best_score = 0
                    
                    for i, config in enumerate(ocr_configs):
                        try:
                            print(f"🔍 Trying OCR config {i+1}: {config}")
                            ocr_text = pytesseract.image_to_string(image, config=config)
                            
                            if ocr_text.strip():
                                score = self._score_w2_ocr_text(ocr_text)
                                print(f"✅ OCR config {i+1} result: {len(ocr_text)} chars, score: {score:.2f}")
                                print(f"📝 Sample: {ocr_text[:100]}...")
                                
                                if score > best_score:
                                    best_ocr_text = ocr_text
                                    best_score = score
                                    print(f"🏆 New best OCR result!")
                            else:
                                print(f"❌ OCR config {i+1}: No text extracted")
                                
                        except Exception as config_e:
                            print(f"❌ OCR config {i+1} failed: {config_e}")
                            continue
                    
                    if best_ocr_text.strip():
                        all_text += "=== OCR EXTRACTION ===\n" + best_ocr_text + "\n"
                        print(f"✅ Best OCR result: {len(best_ocr_text)} characters, score: {best_score:.2f}")
                    else:
                        print("❌ All OCR attempts failed to extract meaningful text")
                else:
                    print("❌ Failed to convert PDF to images")
                
            except Exception as ocr_e:
                print(f"❌ PDF OCR processing failed: {ocr_e}")
                import traceback
                traceback.print_exc()
        
        elif content_type.startswith("image/"):
            print(f"Processing image: {file_path}")
            # Existing image OCR code...
            try:
                image = cv2.imread(file_path)
                if image is not None:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    enhanced = cv2.equalizeHist(gray)
                    denoised = cv2.fastNlMeansDenoising(enhanced)
                    
                    ocr_text = pytesseract.image_to_string(denoised, config='--psm 6 --oem 3')
                    all_text += "=== IMAGE OCR ===\n" + ocr_text + "\n"
                    print(f"✅ Image OCR: {len(ocr_text)} characters")
                else:
                    pil_image = Image.open(file_path)
                    ocr_text = pytesseract.image_to_string(pil_image)
                    all_text += "=== PIL OCR ===\n" + ocr_text + "\n"
                    print(f"✅ PIL OCR: {len(ocr_text)} characters")
                
            except Exception as e:
                print(f"❌ Image OCR failed: {e}")
    
    except Exception as e:
        print(f"❌ Overall text extraction error: {e}")
    
    print(f"📊 Final extracted text length: {len(all_text)} characters")
    if all_text.strip():
        print(f"📝 Text preview: {all_text[:200]}...")
    else:
        print("⚠️  WARNING: No text extracted - will fall back to mock data")
    
    return all_text

def _score_w2_ocr_text(self, text: str) -> float:
    """Score OCR text quality specifically for W-2 forms"""
    if not text.strip():
        return 0.0
    
    score = 0.0
    text_lower = text.lower()
    
    # Check for W-2 specific terms
    w2_terms = ['w-2', 'wage', 'tax', 'statement', 'withheld', 'social security', 'medicare', 'federal', 'employer', 'employee']
    found_terms = sum(1 for term in w2_terms if term in text_lower)
    score += (found_terms / len(w2_terms)) * 0.4
    
    # Check for specific W-2 box indicators
    box_terms = ['wages tips other compensation', 'federal income tax withheld', 'social security wages', 'medicare wages']
    found_boxes = sum(1 for term in box_terms if term in text_lower)
    score += (found_boxes / len(box_terms)) * 0.3
    
    # Check for numbers (tax forms have many numbers)
    numbers = re.findall(r'\d+', text)
    if len(numbers) >= 5:  # Should have multiple numbers
        score += 0.2
    
    # Check for reasonable text length
    if 100 <= len(text) <= 5000:  # Reasonable length for W-2
        score += 0.1
    
    return min(score, 1.0)
