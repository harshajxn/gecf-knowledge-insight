import os
import base64
import json
import sys
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import fitz  # PyMuPDF

load_dotenv()
if os.getenv("GROQ_API_KEY") is None:
    print("WARNING: GROQ_API_KEY not found in environment variables", file=sys.stderr)
    

app = Flask(__name__)
# 100 MB limit for Render
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Enable debug logging
import logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# Usage statistics storage (in production, use a database)
STATS_FILE = "/tmp/usage_stats.json"

def load_stats():
    """Load usage statistics from file"""
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading stats: {e}")
    return {
        "total_visits": 0,
        "total_uploads": 0,
        "total_documents_processed": 0,
        "total_countries_found": 0,
        "visits_by_date": {},
        "uploads_by_date": {},
        "recent_uploads": []
    }

def save_stats(stats):
    """Save usage statistics to file"""
    try:
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        print(f"Error saving stats: {e}")

def track_visit():
    """Track page visit"""
    stats = load_stats()
    stats["total_visits"] += 1
    today = datetime.now().strftime("%Y-%m-%d")
    stats["visits_by_date"][today] = stats["visits_by_date"].get(today, 0) + 1
    save_stats(stats)

def track_upload(num_docs, countries_found, filenames):
    """Track document upload and analysis"""
    stats = load_stats()
    stats["total_uploads"] += 1
    stats["total_documents_processed"] += num_docs
    stats["total_countries_found"] += len(countries_found)
    
    today = datetime.now().strftime("%Y-%m-%d")
    stats["uploads_by_date"][today] = stats["uploads_by_date"].get(today, 0) + 1
    
    # Keep last 10 uploads
    stats["recent_uploads"].insert(0, {
        "timestamp": datetime.now().isoformat(),
        "num_documents": num_docs,
        "countries": list(countries_found),
        "filenames": filenames
    })
    stats["recent_uploads"] = stats["recent_uploads"][:10]
    
    save_stats(stats)

GECF_MEMBER_COUNTRIES = [
    "Algeria", "Bolivia", "Egypt", "Equatorial Guinea", "Iran", "Libya", "Nigeria",
    "Qatar", "Russia", "Trinidad and Tobago", "United Arab Emirates", "UAE", "Venezuela"
]

def extract_images_from_pdf(file_bytes):
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
                    filtered_images.append(base64.b64encode(base_image["image"]).decode("utf-8"))
                except Exception: continue
    except Exception as e: print(f"Error extracting images: {e}")
    return filtered_images

def extract_relevant_text(uploaded_file):
    try:
        file_bytes = uploaded_file.read()
        # Use /tmp directory on Vercel (it's writable)
        temp_dir = "/tmp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, uploaded_file.filename)
        with open(temp_file_path, "wb") as f: f.write(file_bytes)
        pages = PyPDFLoader(temp_file_path).load()
        images = extract_images_from_pdf(file_bytes)
        
        # Clean up temp file
        try:
            os.remove(temp_file_path)
        except:
            pass
            
        document_heading = uploaded_file.filename
        if pages and pages[0].page_content:
            lines = [line.strip() for line in pages[0].page_content.split("\n") if line.strip()]
            if lines: document_heading = " ".join(lines[:2])
        relevant_text, found = "", set()
        for country in GECF_MEMBER_COUNTRIES:
            for page in pages:
                if country.lower() in page.page_content.lower():
                    relevant_text += page.page_content + "\n\n"
                    found.add(country)
        if not relevant_text: return f"No relevant info found in {uploaded_file.filename}.", [], document_heading, []
        return relevant_text, list(found), document_heading, images
    except Exception as e: 
        print(f"Error in extract_relevant_text: {str(e)}")
        return f"Error processing {uploaded_file.filename}: {e}", [], uploaded_file.filename, []

def generate_summary(context: str):
    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
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
    track_visit()
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint to verify deployment"""
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
        all_countries_found = set()
        filenames = []
        
        for file in uploaded_files:
            try:
                filenames.append(file.filename)
                context, countries_found, heading, images = extract_relevant_text(file)
                summary_text = generate_summary(context) if "No relevant info" not in context else context
                mentioned = [c for c in countries_found if c.lower() in summary_text.lower()]
                if "united arab emirates" in summary_text.lower() and "UAE" in countries_found and "UAE" not in mentioned:
                    mentioned.append("UAE")
                
                all_countries_found.update(mentioned)
                
                all_results.append({
                    'fileName': file.filename, 'heading': heading, 'countriesFound': sorted(list(set(mentioned))),
                    'images': images, 'summary': summary_text,
                })
            except Exception as e:
                print(f"Error processing file {file.filename}: {str(e)}")
                all_results.append({
                    'fileName': file.filename,
                    'heading': file.filename,
                    'countriesFound': [],
                    'images': [],
                    'summary': f"Error processing this document: {str(e)}"
                })
        
        # Track this upload
        try:
            track_upload(len(uploaded_files), all_countries_found, filenames)
        except Exception as e:
            print(f"Error tracking upload: {str(e)}")
        
        return jsonify(all_results)
    
    except Exception as e:
        print(f"Error in process_files: {str(e)}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """API endpoint to get usage statistics as JSON"""
    stats = load_stats()
    return jsonify(stats)

@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Dashboard page to view usage statistics"""
    return render_template('stats.html')

if __name__ == '__main__':
    app.run(debug=True)