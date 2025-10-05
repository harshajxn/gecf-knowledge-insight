import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool 
from langchain import hub


# Define the list of GECF member countries for the agent to focus on
GECF_MEMBER_COUNTRIES = [
    "Algeria", "Bolivia", "Egypt", "Equatorial Guinea", "Iran",
    "Libya", "Nigeria", "Qatar", "Russia", "Trinidad and Tobago",
    "United Arab Emirates", "UAE", "Venezuela"
]

# Path to your news document
PDF_PATH = "news_document.pdf"


# --- 2. TOOL FUNCTION DEFINITION ---

# This is now a regular Python function. We will turn it into a tool later.
# --- 2. TOOL FUNCTION DEFINITION (FINAL REFINED VERSION 2.0) ---

def get_news_about_gecf_country(country_name: str) -> str:
    """
    Searches the provided PDF document for SUBSTANTIAL news related to a specific GECF country.
    It filters out pages with very little text (like lists, indexes, or charts) to ensure
    the returned context is high-quality prose.
    """
    print(f"\n--- TOOL CALLED: Searching for news about '{country_name}'... ---")
    
    cleaned_country_name = country_name.strip().lower()

    normalized_countries = [c.strip().lower() for c in GECF_MEMBER_COUNTRIES]
    if cleaned_country_name not in normalized_countries:
        return f"Error: The country '{country_name}' was not found in the predefined list."

    try:
        loader = PyPDFLoader(PDF_PATH)
        pages = loader.load()
        
        relevant_pages_content = []
        # Define a minimum character count to consider a page as "substantial"
        MIN_PAGE_CHARS = 500

        for page in pages:
            page_text_lower = page.page_content.lower()
            
            # --- NEW HEURISTIC: Check for keyword AND minimum page length ---
            if cleaned_country_name in page_text_lower and len(page.page_content) > MIN_PAGE_CHARS:
                relevant_pages_content.append(page.page_content)

        if relevant_pages_content:
            full_context = "\n\n--- [RELEVANT PAGE] ---\n\n".join(relevant_pages_content)
            return f"Success: Found substantial news for {country_name}. The context is: {full_context}"
        else:
            return f"No substantial news mentioning '{country_name}' was found on pages with significant text content."
            
    except FileNotFoundError:
        return f"FATAL ERROR: The file '{PDF_PATH}' was not found."
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

   # --- LOCAL LLM INITIALIZATION WITH OLLAMA ---

# Initialize the ChatOllama model.
# It will automatically connect to the Ollama service running on your machine.
    llm = ChatOllama(
        model="llama3:8b",  # Specify the model you downloaded
        temperature=0       # Set temperature to 0 for more deterministic, factual output
    )

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