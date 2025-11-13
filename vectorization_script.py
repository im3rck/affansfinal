import os
import pandas as pd
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from typing import List
import json
import time

class DataVectorizer:
    """
    Handles vectorization and storage of PDFs and dataset metadata
    Uses Sentence Transformers for embeddings (FREE, no quota limits!)
    """
    
    def __init__(self, gemini_api_key: str = None):
        """Initialize with optional Gemini API key."""
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
        
        # Initialize Sentence Transformer for embeddings (FREE!)
        print("📦 Loading embedding model (this may take a moment first time)...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and efficient
        print("✅ Embedding model loaded!")
        
        # Initialize ChromaDB with persistent storage
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collection for PDFs
        try:
            self.chroma_client.delete_collection("amazon_kb")
            print("🗑️  Cleared existing collection")
        except:
            pass
        
        self.pdf_collection = self.chroma_client.create_collection(
            name="amazon_kb",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.dataset_info = {}
        
    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings using Sentence Transformers (FREE!)."""
        embedding = self.embedding_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundaries
            if end < text_length:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                last_question = chunk.rfind('?')
                break_point = max(last_period, last_newline, last_question)
                
                if break_point > chunk_size * 0.5:  # Only break if we're past halfway
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start = end - overlap
        
        return chunks
    
    def load_and_vectorize_pdfs(self, pdf_paths: List[str]):
        """Load PDFs, create embeddings, and store in ChromaDB."""
        print("="*60)
        print("VECTORIZING PDF DOCUMENTS")
        print("="*60)
        
        total_chunks_all = 0
        
        for pdf_path in pdf_paths:
            try:
                if not os.path.exists(pdf_path):
                    print(f"\n⚠️  File not found: {pdf_path}")
                    continue
                
                print(f"\n📄 Processing: {pdf_path}")
                reader = PdfReader(pdf_path)
                pdf_name = os.path.basename(pdf_path)
                
                total_chunks = 0
                all_chunks = []
                all_metadatas = []
                all_ids = []
                
                # First, collect all chunks
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    
                    if not text.strip():
                        continue
                    
                    # Split into chunks
                    chunks = self.chunk_text(text)
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        doc_id = f"{pdf_name}_page{page_num}_chunk{chunk_idx}"
                        all_chunks.append(chunk)
                        all_metadatas.append({
                            "source": pdf_name,
                            "page": page_num,
                            "chunk": chunk_idx
                        })
                        all_ids.append(doc_id)
                        total_chunks += 1
                
                # Batch generate embeddings (faster!)
                print(f"  🔄 Generating embeddings for {total_chunks} chunks...")
                embeddings = self.embedding_model.encode(
                    all_chunks, 
                    batch_size=32,
                    show_progress_bar=True,
                    convert_to_tensor=False
                )
                
                # Store in ChromaDB in batches
                batch_size = 100
                for i in range(0, len(all_chunks), batch_size):
                    batch_end = min(i + batch_size, len(all_chunks))
                    
                    self.pdf_collection.add(
                        documents=all_chunks[i:batch_end],
                        embeddings=[emb.tolist() for emb in embeddings[i:batch_end]],
                        ids=all_ids[i:batch_end],
                        metadatas=all_metadatas[i:batch_end]
                    )
                
                print(f"  ✅ Processed {len(reader.pages)} pages, {total_chunks} chunks")
                total_chunks_all += total_chunks
                
            except Exception as e:
                print(f"  ✗ Error processing {pdf_path}: {e}")
        
        total_docs = self.pdf_collection.count()
        print(f"\n✅ Total documents in vector DB: {total_docs}")
        print(f"📊 Total chunks created: {total_chunks_all}")
    
    def prepare_dataset_metadata(self, csv_path: str):
        """Load dataset and prepare metadata for queries."""
        print("\n" + "="*60)
        print("PREPARING DATASET METADATA")
        print("="*60)
        
        if not os.path.exists(csv_path):
            print(f"⚠️  CSV file not found: {csv_path}")
            return
        
        df = pd.read_csv(csv_path)
        
        # Generate schema information
        self.dataset_info = {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "sample_rows": df.head(5).to_dict('records'),
            "row_count": len(df),
            "numeric_columns": list(df.select_dtypes(include=['number']).columns),
            "categorical_columns": list(df.select_dtypes(include=['object']).columns)
        }
        
        # Add column descriptions
        column_descriptions = self._generate_column_descriptions(df)
        self.dataset_info["column_descriptions"] = column_descriptions
        
        # Add statistics for numeric columns
        numeric_stats = {}
        for col in self.dataset_info["numeric_columns"]:
            if df[col].notna().sum() > 0:
                numeric_stats[col] = {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "std": float(df[col].std())
                }
        self.dataset_info["numeric_stats"] = numeric_stats
        
        # Add value counts for categorical columns (top 10)
        categorical_stats = {}
        for col in self.dataset_info["categorical_columns"]:
            top_values = df[col].value_counts().head(10).to_dict()
            categorical_stats[col] = {
                "unique_count": int(df[col].nunique()),
                "top_values": {str(k): int(v) for k, v in top_values.items()}
            }
        self.dataset_info["categorical_stats"] = categorical_stats
        
        # Save metadata with UTF-8 encoding
        with open("dataset_metadata.json", "w", encoding='utf-8') as f:
            json.dump(self.dataset_info, f, indent=2, ensure_ascii=False)
        
        # Save the dataframe path
        with open("dataset_path.txt", "w", encoding='utf-8') as f:
            f.write(csv_path)
        
        print(f"\n✅ Dataset metadata prepared:")
        print(f"  - Rows: {self.dataset_info['row_count']:,}")
        print(f"  - Columns: {len(self.dataset_info['columns'])}")
        print(f"  - Numeric columns: {len(self.dataset_info['numeric_columns'])}")
        print(f"  - Categorical columns: {len(self.dataset_info['categorical_columns'])}")
        print(f"\n💾 Metadata saved to: dataset_metadata.json")
    
    def _generate_column_descriptions(self, df: pd.DataFrame) -> dict:
        """Generate human-readable descriptions for columns."""
        descriptions = {}
        
        # Common column descriptions (customize based on your dataset)
        common_descriptions = {
            "product_id": "Unique identifier for each product",
            "product_name": "Name/title of the product",
            "category": "Product category hierarchy",
            "main_category": "Primary product category",
            "sub_category_1": "First level subcategory",
            "sub_category_2": "Second level subcategory",
            "discounted_price": "Current selling price after discount",
            "actual_price": "Original price before discount",
            "discount_percentage": "Percentage discount applied",
            "rating": "Average customer rating (out of 5)",
            "rating_count": "Number of customer ratings",
            "about_product": "Product description and features",
            "user_id": "Unique identifier for reviewers",
            "user_name": "Name of the reviewer",
            "review_id": "Unique identifier for each review",
            "review_title": "Title/summary of the review",
            "review_content": "Detailed review content",
            "savings_amount": "Amount saved compared to actual price",
            "price_range": "Categorized price range (Budget/Premium/etc)",
            "rating_category": "Categorized rating (Excellent/Good/etc)",
            "brand": "Brand name of the product",
            "review_length": "Length of review in characters"
        }
        
        for col in df.columns:
            if col in common_descriptions:
                descriptions[col] = common_descriptions[col]
            else:
                descriptions[col] = f"Data field: {col}"
        
        return descriptions
    
    def save_configuration(self):
        """Save configuration for the frontend."""
        config = {
            "chroma_db_path": "./chroma_db",
            "collection_name": "amazon_kb",
            "dataset_metadata_path": "dataset_metadata.json",
            "dataset_path_file": "dataset_path.txt",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
        }
        
        with open("config.json", "w", encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n💾 Configuration saved to: config.json")


# Main execution
if __name__ == "__main__":
    print("\n" + "="*60)
    print("DATA VECTORIZATION & STORAGE SETUP")
    print("Using Sentence Transformers (FREE, No API Limits!)")
    print("="*60)
    
    # Configuration
    PDF_FILES = [
        "HR & Benefits FAQ.pdf",
        "IT & Tech Support.pdf",
        "Workplace & Operations FAQ.pdf"
    ]
    
    CSV_FILE = "amazon_sales_processed.csv"  # Use the preprocessed file
    
    # Initialize vectorizer (no API key needed!)
    vectorizer = DataVectorizer()
    
    # Process PDFs
    print("\n🔄 Step 1: Vectorizing PDF documents...")
    vectorizer.load_and_vectorize_pdfs(PDF_FILES)
    
    # Process dataset
    print("\n🔄 Step 2: Preparing dataset metadata...")
    vectorizer.prepare_dataset_metadata(CSV_FILE)
    
    # Save configuration
    print("\n🔄 Step 3: Saving configuration...")
    vectorizer.save_configuration()
    
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. ./chroma_db/ - Vector database with PDF embeddings")
    print("  2. dataset_metadata.json - Dataset schema and statistics")
    print("  3. dataset_path.txt - Path to the CSV file")
    print("  4. config.json - Configuration for the frontend")
    print("\n🎉 You can now run the Gradio frontend interface!")
    print("Command: python gradio_frontend.py")
    print("\n💡 Note: This version uses FREE local embeddings - no API quota limits!")