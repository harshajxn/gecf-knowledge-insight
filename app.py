import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq # <-- IMPORT GROQ
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. SETUP: LOAD API KEY, DEFINE GECF COUNTRIES, AND CONSTANTS ---

# Load the Groq API key from the .env file
load_dotenv()

# Check if the API key is loaded and exit if not found
if os.getenv("GROQ_API_KEY") is None:
    print("FATAL ERROR: Groq API key not found.")
    print("Please create a .env file and set GROQ_API_KEY='your_key_here'")
    exit()

GECF_MEMBER_COUNTRIES = [
    "Algeria", "Bolivia", "Egypt", "Equatorial Guinea", "Iran",
    "Libya", "Nigeria", "Qatar", "Russia", "Trinidad and Tobago",
    "United Arab Emirates", "UAE", "Venezuela"
]
PDF_PATH = "news_document.pdf"

# --- 2. THE EXTRACTION FUNCTION (NO LLM, PURE PYTHON) ---

def extract_relevant_text_for_all_countries() -> (str, list):
    """
    Scans the PDF's main body for all GECF countries and extracts relevant pages.
    This function is fast as it does not use an LLM.
    """
    print(f"--- Starting fast extraction process for all {len(GECF_MEMBER_COUNTRIES)} countries... ---")
    
    try:
        loader = PyPDFLoader(PDF_PATH)
        pages = loader.load()
        
        start_page_index, annex_page_index = 0, len(pages)
        for i, page in enumerate(pages):
            page_text_lower = page.page_content.lower()
            if "executive summary" in page_text_lower and start_page_index == 0:
                start_page_index = i
            if "annex" in page_text_lower and "summary table" in page_text_lower:
                annex_page_index = i
                break
        
        print(f"DEBUG: Found content section between page {start_page_index} and {annex_page_index}.")

        all_relevant_text = ""
        countries_found = set()
        
        for country in GECF_MEMBER_COUNTRIES:
            country_lower = country.strip().lower()
            found_this_run = False
            for i in range(start_page_index, annex_page_index):
                page = pages[i]
                if country_lower in page.page_content.lower():
                    if not found_this_run:
                        all_relevant_text += f"\n\n{'='*20}\n--- Relevant text for: {country.upper()} ---\n{'='*20}\n\n"
                        countries_found.add(country)
                        found_this_run = True
                    all_relevant_text += page.page_content + "\n\n--- [End of Page] ---\n"
        
        if not all_relevant_text:
            return "No relevant information found for any GECF countries.", []
        return all_relevant_text, list(countries_found)

    except FileNotFoundError:
        return f"FATAL ERROR: The file '{PDF_PATH}' was not found.", []
    except Exception as e:
        return f"An error occurred during extraction: {e}", []


# --- 3. THE MAIN EXECUTION PIPELINE ---

def main():
    """Main function to run the optimized extraction and summarization pipeline."""
    
    # STEP 1: FAST EXTRACTION
    context, countries_found = extract_relevant_text_for_all_countries()
    
    if not countries_found:
        print(context)
        return
        
    print(f"\n--- Extraction complete. Found news for: {', '.join(countries_found)} ---")
    print("\n--- Now calling the Groq API for a one-shot summary... ---")

    # STEP 2: ONE-SHOT SUMMARIZATION WITH GROQ
    
    # Initialize the Groq LLM
    # We use Llama 3, which is a powerful and very fast model on Groq's hardware.
    llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

    
    # The prompt template remains the same
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are an expert geopolitical energy analyst. Your task is to provide a concise, structured summary
        of the provided text, focusing exclusively on the GECF member countries mentioned.

        Here is the relevant text extracted from a report:
        ---
        {context}
        ---

        Based ONLY on the text provided above, please generate a summary. Organize your summary by country.
        For each country found in the text, provide a bullet point summary of the key information.
        If no substantial news was found for a country, do not mention it.
        Be factual and stick strictly to the provided context.
        """
    )
    
    output_parser = StrOutputParser()
    
    # The chain remains the same
    chain = prompt_template | llm | output_parser
    
    # Invoke the chain with our extracted context
    final_summary = chain.invoke({"context": context})
    
    # STEP 3: DISPLAY THE FINAL RESULT
    print("\n" + "="*50)
    print("--- FINAL SUMMARY ---")
    print("="*50)
    print(final_summary)


if __name__ == "__main__":
    main()