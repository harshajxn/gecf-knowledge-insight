# app.py (Final Version - Aligns Source and Countries on Same Line)

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

# --- GECF Country Lists ---
GECF_MEMBER_COUNTRIES = ["Algeria", "Bolivia", "Egypt", "Equatorial Guinea", "Iran", "Libya", "Nigeria", "Qatar", "Russia", "Trinidad and Tobago", "United Arab Emirates", "UAE", "Venezuela"]
GECF_OBSERVER_COUNTRIES = ["Angola", "Azerbaijan", "Iraq", "Malaysia", "Mauritania", "Mozambique", "Peru", "Senegal"]
ALL_GECF_COUNTRIES = GECF_MEMBER_COUNTRIES + GECF_OBSERVER_COUNTRIES

# --- List of known sources and months, re-ordered for priority ---
KNOWN_SOURCES = ["Rystad Energy", "Enerdata", "Argus", "Wood Mackenzie", "Bloomberg"]
MONTH_NAMES = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]

# --- PDF STYLING CLASS ---
class PDF(FPDF):
    GECF_BLUE = (0, 75, 153)
    TEXT_DARK = (19, 52, 59)
    TEXT_GRAY = (98, 108, 113)
    LINE_COLOR = (220, 220, 220)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.add_font('DejaVu', '', 'fonts/DejaVuSans.ttf', uni=True)
            self.add_font('DejaVu', 'B', 'fonts/DejaVuSans-Bold.ttf', uni=True)
            self.add_font('DejaVu', 'I', 'fonts/DejaVuSans-Oblique.ttf', uni=True)
            self.font_family = 'DejaVu'
        except RuntimeError:
            print("WARNING: DejaVu fonts not found in 'fonts/' directory. Falling back to Helvetica.")
            self.font_family = 'Helvetica'
    
    def header(self):
        self.set_fill_color(*self.GECF_BLUE)
        self.rect(0, 0, self.w, 35, 'F')
        try:
            self.image('static/gecf_logo.png', 15, 8, 20)
        except RuntimeError:
            print("WARNING: Could not find 'static/gecf_logo.png'. Skipping logo in PDF.")
        self.set_text_color(255, 255, 255)
        self.set_font(self.font_family, 'B', 16)
        self.set_xy(40, 9)
        self.cell(0, 8, 'GECF Knowledge Insight Platform')
        self.set_font(self.font_family, '', 10)
        self.set_xy(40, 17)
        self.cell(0, 8, 'Automated News Summary Report')
        self.set_y(12.5)
        self.set_font(self.font_family, '', 9)
        self.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d')}", align='R')
        self.ln(35)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.font_family, 'I', 8)
        self.set_text_color(*self.TEXT_GRAY)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    # <<< --- FINAL FIX: HORIZONTALLY ALIGNED COUNTRIES AND SOURCE --- >>>
    def add_report_entry(self, title, countries, summary, source):
        # --- Title ---
        self.set_font(self.font_family, 'B', 14)
        self.set_text_color(*self.GECF_BLUE)
        self.multi_cell(0, 8, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)

        # --- Combined Countries and Source Line ---
        
        # Part 1: GECF Countries (left-aligned)
        self.set_font(self.font_family, 'B', 9)
        self.set_text_color(*self.TEXT_DARK)
        countries_str = "GECF Countries: " + (", ".join(countries) if countries else "None")
        # Set a fixed width for the countries part
        self.cell(self.w / 2, 8, countries_str)

        # Part 2: Source (right-aligned on the same line)
        self.set_font(self.font_family, 'I', 9)
        self.set_text_color(*self.TEXT_GRAY)
        source_str = ""
        if source and source != "Unknown":
            source_str = f"Source: {source}"
        # Use width=0 to fill the rest of the line, align='R'
        self.cell(0, 8, source_str, align='R', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # --- Summary ---
        self.ln(4) # Add space before summary
        self.set_font(self.font_family, '', 11)
        self.set_text_color(*self.TEXT_DARK)
        self.multi_cell(0, 6, summary, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        # --- Separator Line ---
        self.ln(10)
        self.set_draw_color(*self.LINE_COLOR)
        self.set_line_width(0.5)
        self.line(self.get_x(), self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(10)

# --- Helper Functions and Routes (No changes below) ---
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
        source = "Unknown"
        if pages and pages[0].page_content:
            lines = [line.strip() for line in pages[0].page_content.split("\n") if line.strip()]
            
            if lines:
                document_heading = lines[0]
                if len(lines) > 1:
                    second_line_lower = lines[1].lower()
                    is_source = any(src.lower() in second_line_lower for src in KNOWN_SOURCES)
                    is_date = any(month in second_line_lower for month in MONTH_NAMES)
                    if not is_source and not is_date:
                        document_heading += " " + lines[1]
            
            if len(pages) > 0:
                last_page_text_no_space = pages[-1].page_content.lower().replace(" ", "")
                for src in KNOWN_SOURCES:
                    src_no_space = src.lower().replace(" ", "")
                    if src_no_space in last_page_text_no_space:
                        source = src
                        break
            
            if source == "Unknown":
                first_page_text_no_space = pages[0].page_content.lower().replace(" ", "")
                for src in KNOWN_SOURCES:
                    src_no_space = src.lower().replace(" ", "")
                    if src_no_space in first_page_text_no_space:
                        source = src
                        break

        full_text = "\n\n".join(page.page_content for page in pages)
        
        found = set()
        country_lower = {c.lower() for c in ALL_GECF_COUNTRIES}
        full_text_lower = full_text.lower()
        for country in country_lower:
            if country in full_text_lower:
                found.add(country.capitalize())

        return full_text, list(found), document_heading, images, source
    except Exception as e: 
        print(f"Error in extract_document_data: {str(e)}")
        return f"Error processing {uploaded_file.filename}: {e}", [], uploaded_file.filename, [], "Unknown"

def generate_summary(context: str, countries_found: list):
    try:
        llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0)
        if countries_found:
            template = ("You are an expert geopolitical energy analyst. Directly summarize key insights from the text below in one paragraph, focusing on the role of GECF countries. Do not start with introductory phrases. Avoid lists.\n\nCONTEXT: {context}")
        else:
            template = ("You are an expert analyst. Directly summarize the key insights from the text below in one concise paragraph. Do not start with introductory phrases. Avoid lists.\n\nCONTEXT: {context}")
        prompt = ChatPromptTemplate.from_template(template)
        return (prompt | llm | StrOutputParser()).invoke({"context": context})
    except Exception as e: return f"Could not generate summary: {e}"

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({ 'status': 'ok', 'groq_api_key_set': os.getenv("GROQ_API_KEY") not in [None, ""], 'tmp_writable': os.access('/tmp', os.W_OK) })

@app.route('/process', methods=['POST'])
def process_files():
    try:
        uploaded_files = request.files.getlist('files')
        if not uploaded_files or uploaded_files[0].filename == '':
            return jsonify({'error': "Please upload at least one file."}), 400
        all_results = []
        for file in uploaded_files:
            try:
                context, countries_found, heading, images, source = extract_document_data(file)
                summary_text = generate_summary(context, countries_found)
                mentioned = [c for c in countries_found if c.lower() in summary_text.lower()]
                if "united arab emirates" in summary_text.lower() and "UAE" in countries_found and "UAE" not in mentioned:
                    mentioned.append("UAE")
                all_results.append({ 
                    'fileName': file.filename, 'heading': heading, 
                    'countriesFound': sorted(list(set(mentioned))), 
                    'images': images, 'summary': summary_text, 'source': source
                })
            except Exception as e:
                print(f"Error processing file {file.filename}: {str(e)}")
                all_results.append({ 
                    'fileName': file.filename, 'heading': file.filename, 
                    'countriesFound': [], 'images': [], 
                    'summary': f"Error processing this document: {str(e)}", 'source': 'Unknown'
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
            pdf.add_report_entry(
                report['title'], 
                report['countries'], 
                report['summary'], 
                report.get('source', 'Unknown')
            )
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