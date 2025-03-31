from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import zipfile
import pandas as pd
import tempfile
import os
from typing import Optional
import openai
from dotenv import load_dotenv
import re
import nest_asyncio
import uvicorn
import asyncio

# Load environment variables
load_dotenv()

# Apply nest_asyncio for Jupyter/Spyder compatibility
nest_asyncio.apply()

app = FastAPI()

# --- Secure Configuration ---
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise ValueError("AIPROXY_TOKEN not found in environment variables")

openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1"
openai.api_key = AIPROXY_TOKEN  # Use env variable

# --- File Processing Functions ---
def extract_csv_answer(file_path: str) -> str:
    """Extract answer from CSV file"""
    try:
        df = pd.read_csv(file_path)
        if 'answer' in df.columns:
            return str(df['answer'].iloc[0])
        return str(df.iloc[0, 0])
    except Exception as e:
        raise ValueError(f"CSV Error: {str(e)}")

def process_uploaded_file(file: UploadFile) -> str:
    """Process uploaded file (zip/csv)"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        content = file.file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        if file.filename.endswith('.zip'):
            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                extract_dir = "temp_extract"
                os.makedirs(extract_dir, exist_ok=True)
                zip_ref.extractall(extract_dir)
                
                csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
                if not csv_files:
                    raise ValueError("No CSV in ZIP")
                
                answer = extract_csv_answer(os.path.join(extract_dir, csv_files[0]))
                
                # Cleanup
                for f in os.listdir(extract_dir):
                    os.remove(os.path.join(extract_dir, f))
                os.rmdir(extract_dir)
                return answer
        
        elif file.filename.endswith('.csv'):
            return extract_csv_answer(tmp_path)
        
        else:
            raise ValueError("Unsupported file type")
    
    finally:
        os.unlink(tmp_path)

# --- Enhanced LLM Handler ---
async def get_llm_answer(question: str, context: str = "") -> str:
    """Get answer via AI Proxy"""
    try:
        prompt = f"""Answer concisely with just the required value:
        Question: {question}"""
        
        if context:
            prompt += f"\nContext: {context}"
        
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Only supported model
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        answer = response.choices[0].message.content.strip()
        return re.sub(r'[^a-zA-Z0-9\s]', '', answer)  # Sanitize
    
    except Exception as e:
        error_msg = f"Proxy Error: {str(e)}"
        if "401" in str(e):
            error_msg += " (Invalid Token)"
        elif "model" in str(e):
            error_msg += " (Only gpt-4o-mini supported)"
        raise ValueError(error_msg)

# --- API Endpoint ---
@app.post("/api/")
async def solve_question(
    question: str = Form(...),
    file: Optional[UploadFile] = File(None)
):
    try:
        answer = ""
        file_content = ""
        
        if file:
            file_content = process_uploaded_file(file)
        
        # Question Type Handling
        if "csv" in question.lower() and ("answer" in question.lower() or "value" in question.lower()):
            answer = file_content if file_content else await get_llm_answer(question)
        elif any(word in question.lower() for word in ["calculate", "sum", "multiply"]):
            answer = await get_llm_answer(f"Calculate: {question}")
        else:
            answer = await get_llm_answer(question, file_content)
        
        return {"answer": answer}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# --- Server Setup ---
def run_server():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.serve())

if __name__ == "__main__":
    run_server()