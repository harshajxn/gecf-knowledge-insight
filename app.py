import os
import base64
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import fitz  # PyMuPDF

load_dotenv()
if os.getenv("GROQ_API_KEY") is None:
    raise Exception("FATAL ERROR: GROQ_API_KEY not found in .env file.")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024 # 200 MB limit per request

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
        temp_dir = "temp"
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
        relevant_text, found = "", set()
        for country in GECF_MEMBER_COUNTRIES:
            for page in pages:
                if country.lower() in page.page_content.lower():
                    relevant_text += page.page_content + "\n\n"
                    found.add(country)
        if not relevant_text: return f"No relevant info found in {uploaded_file.filename}.", [], document_heading, []
        return relevant_text, list(found), document_heading, images
    except Exception as e: return f"Error processing {uploaded_file.filename}: {e}", [], uploaded_file.filename, []

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
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_files():
    uploaded_files = request.files.getlist('files')
    if not uploaded_files or uploaded_files[0].filename == '':
        return jsonify({'error': "Please upload at least one file."}), 400

    all_results = []
    for file in uploaded_files:
        context, countries_found, heading, images = extract_relevant_text(file)
        summary_text = generate_summary(context) if "No relevant info" not in context else context
        mentioned = [c for c in countries_found if c.lower() in summary_text.lower()]
        if "united arab emirates" in summary_text.lower() and "UAE" in countries_found and "UAE" not in mentioned:
            mentioned.append("UAE")
        all_results.append({
            'fileName': file.filename, 'heading': heading, 'countriesFound': sorted(list(set(mentioned))),
            'images': images, 'summary': summary_text,
        })
    return jsonify(all_results)

if __name__ == '__main__':
    app.run(debug=True)