import os
import base64
import sys
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import fitz  # PyMuPDF
from PIL import Image
import io

# Load environment variables from .env file
load_dotenv()

# Check for API key on startup
if os.getenv("GROQ_API_KEY") is None:
    print("WARNING: GROQ_API_KEY not found in environment variables", file=sys.stderr)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 # Set a high limit for direct uploads

# Enable logging
import logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

GECF_MEMBER_COUNTRIES = [
    "Algeria", "Bolivia", "Egypt", "Equatorial Guinea", "Iran", "Libya", "Nigeria",
    "Qatar", "Russia", "Trinidad and Tobago", "United Arab Emirates", "UAE", "Venezuela"
]

# --- Helper function to resize images ---
def resize_and_encode_image(image_bytes, max_width=800):
    """
    Resizes an image to a max width, converts it to a web-friendly JPEG,
    and returns a Base64 encoded string. This significantly reduces data size.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.size[0] > max_width:
            w_percent = (max_width / float(img.size[0]))
            h_size = int((float(img.size[1]) * float(w_percent)))
            img = img.resize((max_width, h_size), Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=85)
        
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error resizing image: {e}")
        return None

def extract_images_from_pdf(file_bytes):
    """Extracts and resizes images to create smaller thumbnails."""
    MIN_WIDTH, MIN_HEIGHT, HEADER_FOOTER_MARGIN = 100, 100, 0.15
    filtered_images = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            image_list = page.get_images(full=True)
            if not image_list: continue
            header_boundary = page.rect.height * HEADER_FOOTER_MARGIN
            footer_boundary = page.rect.height * (1 - HEADER_FOOTER_MARGIN)
            for img_info in image_list:
                try:
                    base_image = doc.extract_image(img_info[0])
                    if base_image["width"] < MIN_WIDTH or base_image["height"] < MIN_HEIGHT: continue
                    bbox = page.get_image_bbox(img_info)
                    if bbox.y1 < header_boundary or bbox.y0 > footer_boundary: continue
                    
                    resized_image_b64 = resize_and_encode_image(base_image["image"])
                    if resized_image_b64:
                        filtered_images.append(resized_image_b64)
                        
                except Exception: continue
    except Exception as e: print(f"Error extracting images: {e}")
    return filtered_images

def extract_relevant_text(uploaded_file):
    try:
        file_bytes = uploaded_file.read()
        temp_dir = "/tmp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, uploaded_file.filename)
        with open(temp_file_path, "wb") as f: f.write(file_bytes)
        
        pages = PyPDFLoader(temp_file_path).load()
        images = extract_images_from_pdf(file_bytes)
        
        os.remove(temp_file_path) # Clean up temp file
            
        document_heading = uploaded_file.filename
        if pages and pages[0].page_content:
            lines = [line.strip() for line in pages[0].page_content.split("\n") if line.strip()]
            if lines: document_heading = " ".join(lines[:2])

        # Optimized loop for text extraction
        relevant_text_parts = []
        found = set()
        country_lower = {c.lower() for c in GECF_MEMBER_COUNTRIES}
        for page in pages:
            page_content_lower = page.page_content.lower()
            found_on_page = False
            for country in country_lower:
                if country in page_content_lower:
                    found.add(country.capitalize())
                    found_on_page = True
            if found_on_page:
                relevant_text_parts.append(page.page_content)
        relevant_text = "\n\n".join(relevant_text_parts)

        if not relevant_text: return f"No relevant info found in {uploaded_file.filename}.", [], document_heading, []
        return relevant_text, list(found), document_heading, images
    except Exception as e: 
        print(f"Error in extract_relevant_text: {str(e)}")
        return f"Error processing {uploaded_file.filename}: {e}", [], uploaded_file.filename, []

def generate_summary(context: str):
    try:
        llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)
        prompt = ChatPromptTemplate.from_template(
            "You are an expert geopolitical energy analyst. "
            "Directly summarize key insights from the text below in one paragraph, focusing on GECF member countries. "
            "Do not start with introductory phrases. Avoid lists.\n\n"
            "CONTEXT: {context}"
        )
        return (prompt | llm | StrOutputParser()).invoke({"context": context})
    except Exception as e: return f"Could not generate summary: {e}"

@app.route('/', methods=['GET'])
def home():
    # No tracking needed here
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'groq_api_key_set': os.getenv("GROQ_API_KEY") is not None,
        'tmp_writable': os.access('/tmp', os.W_OK)
    })

@app.route('/process', methods=['POST'])
def process_files():
    try:
        uploaded_files = request.files.getlist('files')
        if not uploaded_files or uploaded_files[0].filename == '':
            return jsonify({'error': "Please upload at least one file."}), 400

        all_results = []
        
        for file in uploaded_files:
            try:
                context, countries_found, heading, images = extract_relevant_text(file)
                summary_text = generate_summary(context) if "No relevant info" not in context else context
                mentioned = [c for c in countries_found if c.lower() in summary_text.lower()]
                if "united arab emirates" in summary_text.lower() and "UAE" in countries_found and "UAE" not in mentioned:
                    mentioned.append("UAE")
                
                all_results.append({
                    'fileName': file.filename, 'heading': heading, 'countriesFound': sorted(list(set(mentioned))),
                    'images': images, 'summary': summary_text,
                })
            except Exception as e:
                print(f"Error processing file {file.filename}: {str(e)}")
                all_results.append({
                    'fileName': file.filename, 'heading': file.filename,
                    'countriesFound': [], 'images': [],
                    'summary': f"Error processing this document: {str(e)}"
                })
        
        # No tracking needed here
        
        return jsonify(all_results)
    
    except Exception as e:
        print(f"Error in process_files: {str(e)}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)