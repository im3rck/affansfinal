import os
import pandas as pd
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
import google.generativeai as genai
from typing import List, Dict
import json

class AmazonRAGChatbot:
    def __init__(self, gemini_api_key: str):
        """Initialize the RAG chatbot with Gemini API and ChromaDB."""
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        # Create collection for PDF documents
        try:
            self.chroma_client.delete_collection("amazon_kb")
        except:
            pass
        self.collection = self.chroma_client.create_collection(
            name="amazon_kb",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.df = None
        self.dataset_schema = None
        
    def load_pdfs(self, pdf_paths: List[str]):
        """Load and process PDF documents into ChromaDB."""
        print("Loading PDFs into vector database...")
        
        for pdf_path in pdf_paths:
            try:
                reader = PdfReader(pdf_path)
                pdf_name = os.path.basename(pdf_path)
                
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    
                    # Split into chunks (simple chunking by paragraphs)
                    chunks = self._chunk_text(text)
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        if chunk.strip():
                            doc_id = f"{pdf_name}_page{page_num}_chunk{chunk_idx}"
                            self.collection.add(
                                documents=[chunk],
                                ids=[doc_id],
                                metadatas=[{
                                    "source": pdf_name,
                                    "page": page_num,
                                    "chunk": chunk_idx
                                }]
                            )
                
                print(f"✓ Loaded {pdf_name}")
            except Exception as e:
                print(f"✗ Error loading {pdf_path}: {e}")
        
        print(f"Total documents in vector DB: {self.collection.count()}")
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += chunk_size - overlap
        
        return chunks
    
    def load_dataset(self, csv_path: str):
        """Load the sales dataset."""
        print(f"Loading dataset from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # Generate schema information
        self.dataset_schema = {
            "columns": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "sample_rows": self.df.head(3).to_dict('records'),
            "row_count": len(self.df)
        }
        
        print(f"✓ Loaded dataset with {len(self.df)} rows and {len(self.df.columns)} columns")
        print(f"Columns: {', '.join(self.df.columns)}")
    
    def _classify_query(self, query: str) -> str:
        """Classify whether query is about documents or dataset."""
        classification_prompt = f"""Classify the following query into one of two categories:
1. "DOCUMENT" - if the query is asking about company policies, procedures, knowledge base, documentation, or general information
2. "DATASET" - if the query is asking for data analysis, statistics, numbers, sales data, or requires querying a database

Query: {query}

Respond with only "DOCUMENT" or "DATASET"."""

        response = self.model.generate_content(classification_prompt)
        classification = response.text.strip().upper()
        
        return "DOCUMENT" if "DOCUMENT" in classification else "DATASET"
    
    def _query_documents(self, query: str, n_results: int = 3) -> str:
        """Query the vector database for relevant document chunks."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        if not results['documents'][0]:
            return "No relevant information found in the knowledge base."
        
        # Combine retrieved chunks
        context = "\n\n".join(results['documents'][0])
        
        # Generate answer using Gemini
        prompt = f"""Based on the following context from the company knowledge base, answer the user's question.
If the answer is not in the context, say so.

Context:
{context}

Question: {query}

Answer:"""

        response = self.model.generate_content(prompt)
        return response.text
    
    def _query_dataset(self, query: str) -> Dict:
        """Generate and execute SQL-like query on the dataset."""
        # Generate SQL query using Gemini
        sql_prompt = f"""Given this dataset schema:
Columns: {', '.join(self.dataset_schema['columns'])}
Data types: {json.dumps(self.dataset_schema['dtypes'], indent=2)}
Sample data: {json.dumps(self.dataset_schema['sample_rows'], indent=2)}

Generate a pandas query or SQL-like description to answer this question: {query}

Provide:
1. A SQL-style query (even though we're using pandas)
2. The actual pandas code to execute

Format your response as:
SQL: <sql query here>
PANDAS: <pandas code here>

Use 'df' as the dataframe variable name."""

        response = self.model.generate_content(sql_prompt)
        response_text = response.text
        
        # Extract SQL and pandas code
        sql_query = ""
        pandas_code = ""
        
        lines = response_text.split('\n')
        capture_sql = False
        capture_pandas = False
        
        for line in lines:
            if 'SQL:' in line:
                capture_sql = True
                sql_query = line.split('SQL:', 1)[1].strip()
                continue
            elif 'PANDAS:' in line:
                capture_sql = False
                capture_pandas = True
                pandas_code = line.split('PANDAS:', 1)[1].strip()
                continue
            
            if capture_sql:
                sql_query += " " + line.strip()
            elif capture_pandas:
                pandas_code += "\n" + line
        
        # Clean up code
        pandas_code = pandas_code.strip()
        if pandas_code.startswith('```'):
            pandas_code = '\n'.join(pandas_code.split('\n')[1:-1])
        
        # Execute pandas code safely
        result = None
        error = None
        
        try:
            # Create a safe execution environment
            local_vars = {'df': self.df, 'pd': pd}
            exec(pandas_code, {'__builtins__': {}}, local_vars)
            
            # Get the result (look for 'result' variable or last dataframe operation)
            if 'result' in local_vars:
                result = local_vars['result']
            else:
                # Try to find any dataframe in local_vars
                for var_name, var_value in local_vars.items():
                    if isinstance(var_value, (pd.DataFrame, pd.Series, int, float, str)):
                        result = var_value
                        break
        except Exception as e:
            error = str(e)
        
        return {
            "sql_query": sql_query,
            "pandas_code": pandas_code,
            "result": result,
            "error": error
        }
    
    def chat(self, query: str) -> Dict:
        """Main chat interface."""
        print(f"\n🤖 Processing query: {query}")
        
        # Classify the query
        query_type = self._classify_query(query)
        print(f"📋 Query type: {query_type}")
        
        if query_type == "DOCUMENT":
            answer = self._query_documents(query)
            return {
                "type": "document",
                "query": query,
                "answer": answer
            }
        else:
            result = self._query_dataset(query)
            
            # Format the result
            answer = f"SQL Query: {result['sql_query']}\n\n"
            answer += f"Pandas Code:\n{result['pandas_code']}\n\n"
            
            if result['error']:
                answer += f"Error: {result['error']}"
            else:
                answer += f"Result:\n{result['result']}"
            
            return {
                "type": "dataset",
                "query": query,
                "sql_query": result['sql_query'],
                "pandas_code": result['pandas_code'],
                "result": result['result'],
                "error": result['error'],
                "answer": answer
            }


# Example usage
if __name__ == "__main__":
    # Initialize chatbot
    GEMINI_API_KEY = "AIzaSyAKrO_ucEVGefsHBn7wcP_IMSBS3yqmA9E" 
    chatbot = AmazonRAGChatbot(GEMINI_API_KEY)
    
    # Load PDFs (replace with your actual PDF paths)
    pdf_files = [
        "HR & Benefits FAQ.pdf",
        "IT & Tech Support.pdf",
        "Workplace & Operations FAQ.pdf"
    ]
    chatbot.load_pdfs(pdf_files)
    
    # Load dataset (replace with your actual CSV path)
    chatbot.load_dataset("amazon_sales.csv")
    
    # Example queries
    print("\n" + "="*60)
    print("CHATBOT READY - Example Queries")
    print("="*60)
    
    # Document query example
    response1 = chatbot.chat("What is the return policy?")
    print(f"\n✅ Answer: {response1['answer']}")
    
    # Dataset query example
    response2 = chatbot.chat("Show me total sales by category")
    print(f"\n✅ Answer: {response2['answer']}")
    
    # Interactive mode
    print("\n" + "="*60)
    print("Interactive Mode (type 'quit' to exit)")
    print("="*60)
    
    while True:
        user_query = input("\n💬 You: ").strip()
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_query:
            continue
        
        response = chatbot.chat(user_query)
        print(f"\n🤖 Assistant: {response['answer']}")