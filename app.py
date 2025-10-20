import os
import base64
import sys
from flask import Flask, render_template, request, jsonify, make_response
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import fitz
from PIL import Image
import io
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from datetime import datetime

# Load environment variables
load_dotenv()

# Check for API key
if os.getenv("GROQ_API_KEY") is None:
    print("WARNING: GROQ_API_KEY not found in environment variables", file=sys.stderr)

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# --- Logging ---
import logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# --- UPDATED: Added Observer Countries and a combined list ---
GECF_MEMBER_COUNTRIES = [
    "Algeria", "Bolivia", "Egypt", "Equatorial Guinea", "Iran", "Libya", "Nigeria",
    "Qatar", "Russia", "Trinidad and Tobago", "United Arab Emirates", "UAE", "Venezuela"
]

GECF_OBSERVER_COUNTRIES = [
    "Angola", "Azerbaijan", "Iraq", "Malaysia", "Mauritania", "Mozambique", "Peru", "Senegal"
]

ALL_GECF_COUNTRIES = GECF_MEMBER_COUNTRIES + GECF_OBSERVER_COUNTRIES

# --- PDF STYLING CLASS (No changes here) ---
class PDF(FPDF):
    GECF_BLUE = (0, 75, 153)
    TEXT_DARK = (19, 52, 59)
    TEXT_GRAY = (98, 108, 113)
    LINE_COLOR = (220, 220, 220)
    
    def header(self):
        self.set_fill_color(*self.GECF_BLUE)
        self.rect(0, 0, self.w, 35, 'F')
        try:
            self.image('static/gecf_logo.png', 15, 8, 20)
        except RuntimeError:
            print("WARNING: Could not find 'static/gecf_logo.png'. Skipping logo in PDF.")
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 16)
        self.set_xy(40, 9)
        self.cell(0, 8, 'GECF Knowledge Insight Platform')
        self.set_font('Helvetica', '', 10)
        self.set_xy(40, 17)
        self.cell(0, 8, 'Automated News Summary Report')
        self.set_y(12.5)
        self.set_font('Helvetica', '', 9)
        self.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d')}", align='R')
        self.ln(35)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(*self.TEXT_GRAY)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def add_report_entry(self, title, countries, summary):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(*self.GECF_BLUE)
        self.multi_cell(0, 8, title.encode('latin-1', 'replace').decode('latin-1'))
        self.ln(2)
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(*self.TEXT_DARK)
        countries_str = "GECF Countries: " + (", ".join(countries) if countries else "None")
        self.cell(0, 8, countries_str.encode('latin-1', 'replace').decode('latin-1'))
        self.ln(8)
        self.set_font('Helvetica', '', 11)
        self.set_text_color(*self.TEXT_DARK)
        self.multi_cell(0, 6, summary.encode('latin-1', 'replace').decode('latin-1'))
        self.ln(10)
        self.set_draw_color(*self.LINE_COLOR)
        self.set_line_width(0.5)
        self.line(self.get_x(), self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(10)

# --- Helper Functions and Routes ---
def resize_and_encode_image(image_bytes, max_width=800):
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
    # ... (No changes to this function)
    return []

# --- UPDATED: This function now returns the FULL text and a list of any GECF countries found ---
def extract_document_data(uploaded_file):
    try:
        file_bytes = uploaded_file.read()
        temp_dir = "/tmp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, uploaded_file.filename)
        with open(temp_file_path, "wb") as f: f.write(file_bytes)
        
        pages = PyPDFLoader(temp_file_path).load()
        images = extract_images_from_pdf(file_bytes)
        os.remove(temp_file_path)
            
        document_heading = uploaded_file.filename
        if pages and pages[0].page_content:
            lines = [line.strip() for line in pages[0].page_content.split("\n") if line.strip()]
            if lines: document_heading = " ".join(lines[:2])

        full_text = "\n\n".join(page.page_content for page in pages)
        
        found = set()
        country_lower = {c.lower() for c in ALL_GECF_COUNTRIES}
        full_text_lower = full_text.lower()
        for country in country_lower:
            if country in full_text_lower:
                found.add(country.capitalize())

        return full_text, list(found), document_heading, images
    except Exception as e: 
        print(f"Error in extract_document_data: {str(e)}")
        return f"Error processing {uploaded_file.filename}: {e}", [], uploaded_file.filename, []

# --- UPDATED: This function now uses a dynamic prompt ---
def generate_summary(context: str, countries_found: list):
    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        
        if countries_found:
            template = (
                "You are an expert geopolitical energy analyst. "
                "Directly summarize key insights from the text below in one paragraph, focusing on the role of GECF countries. "
                "Do not start with introductory phrases. Avoid lists.\n\n"
                "CONTEXT: {context}"
            )
        else:
            template = (
                "You are an expert analyst. "
                "Directly summarize the key insights from the text below in one concise paragraph. "
                "Do not start with introductory phrases. Avoid lists.\n\n"
                "CONTEXT: {context}"
            )
        
        prompt = ChatPromptTemplate.from_template(template)
        return (prompt | llm | StrOutputParser()).invoke({"context": context})
    except Exception as e: return f"Could not generate summary: {e}"

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({ 'status': 'ok', 'groq_api_key_set': os.getenv("GROQ_API_KEY") not in [None, ""], 'tmp_writable': os.access('/tmp', os.W_OK) })

# --- UPDATED: Main processing logic now summarizes every file ---
@app.route('/process', methods=['POST'])
def process_files():
    try:
        uploaded_files = request.files.getlist('files')
        if not uploaded_files or uploaded_files[0].filename == '':
            return jsonify({'error': "Please upload at least one file."}), 400

        all_results = []
        for file in uploaded_files:
            try:
                context, countries_found, heading, images = extract_document_data(file)
                
                # Always generate a summary, but pass in the list of countries found
                # to determine which prompt to use.
                summary_text = generate_summary(context, countries_found)
                
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
                    'fileName': file.filename, 'heading': file.filename, 'countriesFound': [], 'images': [],
                    'summary': f"Error processing this document: {str(e)}"
                })
        
        return jsonify(all_results)
    
    except Exception as e:
        print(f"Error in process_files: {str(e)}")
        return jsonify({'error': f"Server error: {str(e)}"}), 500

@app.route('/generate-pdf', methods=['POST'])
def generate_pdf_report():
    try:
        selected_reports = request.get_json()
        if not selected_reports:
            return jsonify({'error': 'No reports selected'}), 400
        
        pdf = PDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        for report in selected_reports:
            pdf.add_report_entry(report['title'], report['countries'], report['summary'])
        
        pdf_bytes = bytes(pdf.output())
        
        response = make_response(pdf_bytes)
        response.headers.set('Content-Type', 'application/pdf')
        response.headers.set('Content-Disposition', 'attachment', filename='GECF_News_Report.pdf')
        return response

    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return jsonify({'error': f"Server error while generating PDF: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)