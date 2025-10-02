import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool # <-- IMPORT THE Tool CLASS
from langchain import hub

# --- 1. SETUP: LOAD API KEY AND DEFINE GECF COUNTRIES ---

# Load the Hugging Face API token from the .env file
load_dotenv()

# Check if the API key is loaded and exit if not found
if os.getenv("HUGGINGFACEHUB_API_TOKEN") is None:
    print("FATAL ERROR: Hugging Face API token not found.")
    print("Please create a .env file and set HUGGINGFACEHUB_API_TOKEN='your_token_here'")
    exit()

# Define the list of GECF member countries for the agent to focus on
GECF_MEMBER_COUNTRIES = [
    "Algeria", "Bolivia", "Egypt", "Equatorial Guinea", "Iran",
    "Libya", "Nigeria", "Qatar", "Russia", "Trinidad and Tobago",
    "United Arab Emirates", "UAE", "Venezuela"
]

# Path to your news document
PDF_PATH = "news_document.pdf"


# --- 2. TOOL FUNCTION DEFINITION (WITHOUT @tool DECORATOR) ---

# This is now a regular Python function. We will turn it into a tool later.
def get_news_about_gecf_country(country_name: str) -> str:
    """
    Searches the provided PDF document for news related to a specific GECF country.
    Returns a text snippet if the country is mentioned, otherwise indicates that
    no specific news was found for that country. This tool is essential for gathering
    information before summarizing.
    """
    print(f"\n--- TOOL CALLED: Searching for news about '{country_name}'... ---")
    
    # Clean the input from the agent to remove potential whitespace or newlines
    cleaned_country_name = country_name.strip().lower()

    # Create a normalized list of countries for comparison
    normalized_countries = [c.strip().lower() for c in GECF_MEMBER_COUNTRIES]

    # Sanity check to ensure the agent is asking for a valid country
    if cleaned_country_name not in normalized_countries:
        # Provide a more helpful error message
        return f"Error: The country '{country_name}' was not found in the predefined list of GECF countries. Please only use countries from the list."

    try:
        # Load the PDF document
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        full_text = " ".join([page.page_content for page in documents])

        # Simple text search for the country name (using the cleaned name)
        if cleaned_country_name in full_text.lower():
            # For efficiency, we return only the first 3500 characters as context.
            return f"Success: Found relevant news for {country_name}. The context is: {full_text[:3500]}"
        else:
            return f"No specific news mentioning '{country_name}' was found in the document."
            
    except FileNotFoundError:
        return f"FATAL ERROR: The file '{PDF_PATH}' was not found. Please make sure it is in the same directory as this script."
    except Exception as e:
        return f"An error occurred while processing the PDF: {e}"

# --- 3. AGENT AND LLM INITIALIZATION ---

def main():
    """Main function to set up and run the agent."""

    print("Initializing GECF News Agent...")

    # --- CORRECT WAY TO CREATE THE TOOL LIST ---
    # We now explicitly create a Tool object from our function.
    tools = [
        Tool(
            name="get_news_about_gecf_country",
            func=get_news_about_gecf_country,
            description="Searches the provided PDF document for news related to a specific GECF country. You must use this tool for every country in the list before answering."
        )
    ]

    # --- CORRECT LLM INITIALIZATION (FINAL VERSION) ---

    # Step A: Define the basic LLM connection using HuggingFaceEndpoint.
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.1,
        max_new_tokens=1024,
        return_full_text=False
    )

    # Step B: Wrap the basic LLM connection in the ChatHuggingFace wrapper.
    llm = ChatHuggingFace(llm=llm_endpoint)

    # --- AGENT AND EXECUTOR SETUP ---

    # Get the ReAct agent prompt template from LangChain Hub.
    prompt = hub.pull("hwchase17/react")
    
    # Create the agent by binding the LLM, the available tools, and the prompt together
    agent = create_react_agent(llm, tools, prompt)
    
    # Create the Agent Executor, which is the runtime environment that makes the agent work.
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=25  )

    # --- 4. RUN THE AGENT WITH ITS TASK ---

    print("\nAgent is ready. Starting the summarization task...")
    
    task_prompt = f"""
    Your final answer MUST be a well-structured summary of the news found in the document
    related ONLY to the GECF member countries.
    The list of countries to check is: {', '.join(GECF_MEMBER_COUNTRIES)}.
    
    You MUST follow this plan:
    1. For each country in the provided list, you absolutely MUST use the 'get_news_about_gecf_country' tool one time.
    2. Review the output from the tool for that country.
    3. After checking ALL the countries, compile the information you found into a final, consolidated summary.
    4. For countries where no news was found, explicitly state that in your final summary.
    5. Do not include information about any other country or topic. Do not invent information.
    Begin!
    """
    
    # Invoke the agent to start the task
    response = agent_executor.invoke({"input": task_prompt})

    print("\n" + "="*50)
    print("--- AGENT'S FINAL SUMMARY ---")
    print("="*50)
    print(response['output'])


if __name__ == "__main__":
    main()