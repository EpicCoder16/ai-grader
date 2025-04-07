from fastapi import FastAPI, UploadFile, File
import shutil
import docx
import pdfplumber
import os
from sentence_transformers import SentenceTransformer, util

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify the exact origins you want to allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Global variable to store the answer key text
answer_key_text = None

# Function to extract text from a .docx file
def extract_text_from_docx(file_path: str):
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Function to extract text from a .pdf file
def extract_text_from_pdf(file_path: str):
    with pdfplumber.open(file_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Endpoint to upload the answer key
@app.post("/upload_answer_key/")
async def upload_answer_key(file: UploadFile = File(...)):
    global answer_key_text  # To update the answer key globally
    
    # Save the file to a temporary location
    file_location = f"uploads/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Extract text based on file type
    if file.filename.endswith('.docx'):
        answer_key_text = extract_text_from_docx(file_location)
    elif file.filename.endswith('.pdf'):
        answer_key_text = extract_text_from_pdf(file_location)
    else:
        return {"error": "Unsupported file type. Please upload a .docx or .pdf file."}
    
    return {"filename": file.filename, "message": "Answer key uploaded successfully!"}

@app.get("/")
def read_root():
    return {"message": "Hello, AI Grader!"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Check if the answer key has been uploaded
    if not answer_key_text:
        return {"error": "Answer key is not uploaded yet. Please upload the answer key first."}
    
    # Save the student file to a temporary location
    file_location = f"uploads/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Extract text based on file type
    if file.filename.endswith('.docx'):
        extracted_text = extract_text_from_docx(file_location)
    elif file.filename.endswith('.pdf'):
        extracted_text = extract_text_from_pdf(file_location)
    else:
        return {"error": "Unsupported file type. Please upload a .docx or .pdf file."}

    # Compare the extracted text with the answer key using Sentence-BERT
    comparison_result = compare_with_answer_key(extracted_text, answer_key_text)

    print(f"Comparison result: {comparison_result}")

    return {"filename": file.filename, "extracted_text": extracted_text, "comparison_result": comparison_result}

# Function to compare extracted text with answer key using Sentence-BERT
def compare_with_answer_key(extracted_text: str, answer_key: str):
    # Encode the extracted text and the answer key into embeddings
    extracted_embedding = model.encode(extracted_text, convert_to_tensor=True)
    answer_key_embedding = model.encode(answer_key, convert_to_tensor=True)
    
    # Compute the cosine similarity between the two embeddings
    cosine_similarity = util.pytorch_cos_sim(extracted_embedding, answer_key_embedding)

    # Return the similarity score
    similarity_score = cosine_similarity.item()  # Extract the value from the tensor
    return {"similarity_score": similarity_score, "message": "Comparison complete."}
