# RAG-based Query Resolver

This project implements a **Retrieval-Augmented Generation (RAG) pipeline** to answer queries based on user-provided text files. Users can add documents, set up the Google Gemini API, and run the application to ask questions based on the content of the added files.

---
# Features
Add your own .txt files to the data/ folder to create a custom knowledge base.
Uses the Google Gemini API for query resolution.
Answers queries based on the content of your text files.
Easy-to-run Python application.

## Quick Start

Follow these three simple steps to start querying your documents:

```bash
# 1. Clone the repository
git clone <repo-url>
cd <repo-directory>

# 2. Make a virtual environment and install all the required dependencies
pip intall -r requirements.txt

# 3. Add your .txt files to the 'data/' folder

# 4. Set your Google Gemini API key in .env file as shown in .env_example
export GOOGLE_GEMINI_API_KEY="your_api_key_here"  # Linux/macOS
setx GOOGLE_GEMINI_API_KEY "your_api_key_here"    # Windows PowerShell


# 5. Run the app
python src/app.py
