### Bank Customer Service Chatbot (RAG-Based)

Welcome to the **Bank Customer Service Chatbot** project! This chatbot is built using Retrieval-Augmented Generation (RAG) and leverages a knowledge base (`bank.pdf`) to provide accurate responses to customer queries. Below are the steps to set up and run the project on your local machine.

Prerequisites

Before starting, ensure that the following tools are installed on your PC:

    1. Python (Version 3.12.6):  
        Download Python from https://www.python.org/downloads/release/python-3126/ and install it.

    2. Ollama:  
        Download and install Ollama from https://ollama.com/download.

    3. Ensure you have the following files in your project directory:
        - `bank.pdf` (Knowledge Base)
        - `testset.csv.txt` (Test Dataset)
        - `requirements.txt` (Dependencies)

Setup Instructions

Step 1: Create a Virtual Environment
        1. Open a terminal (Command Prompt or PowerShell).
        2. Navigate to your project directory.
              cd path/to/your/project
        3. Create a virtual environment named `.venv`:
                python -m venv .venv
        4. Activate the virtual environment:
                - On Windows: .venv\Scripts\activate
                - On macOS/Linux: source .venv/bin/activate

Step 2: Install Dependencies
        1. Install all required libraries and dependencies listed in `requirements.txt`: pip install -r requirements.txt
   

Step 3: Install Llama3.2 Model Using Ollama
        1. Open a terminal and start Ollama: ollama
   
        2. Pull the Llama 3.2 model: ollama pull llama3.2
   
        3. Run the Llama 3.2 model: ollama run llama3.2

Step 4: Prepare the Knowledge Base
    Ensure the following files are present in your project directory:
        - `bank.pdf`: The knowledge base used by the chatbot.
        - `testset.csv.txt`: The dataset for automation testing.

Step 5: Create the Vector Database
    Run the `create_database.py` script to create the vector database: python create_database.py

Step 6: Run the Chatbot
    1. Start the chatbot using Streamlit: streamlit run runbot.py

    2. A new browser window will open, allowing you to interact with the chatbot.

Step 7: Run Automation Testing
    1. Execute the automation testing script using Streamlit: streamlit run automation_test.py
    2. Review the test results in the browser window.

Additional Notes:
    - Virtual Environment: Always activate the virtual environment before running any scripts.
    - Streamlit: Ensure Streamlit is installed (it should be included in `requirements.txt`). If not, install it manually: pip install streamlit
    - Ollama: Keep the Ollama service running in the background while using the chatbot.

Troubleshooting:
    - Missing Files: Ensure `bank.pdf`, `testset.csv.txt`, and `requirements.txt` are in the project directory.
    - Permission Denied: Use `--user` with `pip` if you encounter permission issues: pip install -r requirements.txt --user
    - Outdated Pip: Update `pip` if needed: pip install --upgrade pip

Contact
For any questions or issues, feel free to reach out to the project maintainer:
    - Email: muzankibu977@gmail.com
    - GitHub: 

Thank you for using the **Bank Customer Service Chatbot**! We hope this tool helps streamline customer support and enhances user experience.
