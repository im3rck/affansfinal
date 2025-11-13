import gradio as gr
import pandas as pd
import chromadb
import google.generativeai as genai
import json
from typing import List, Dict, Tuple
import os

class SmartAmazonChatbot:
    
    def __init__(self, gemini_api_key: str):
        """Initialize the chatbot with Gemini API and load stored data."""
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.embedding_model = 'models/embedding-001'
        
        # Load configuration
        with open("config.json", "r", encoding='utf-8') as f:
            config = json.load(f)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=config["chroma_db_path"])
        self.pdf_collection = self.chroma_client.get_collection(config["collection_name"])
        
        # Load dataset
        with open(config["dataset_path_file"], "r", encoding='utf-8') as f:
            dataset_path = f.read().strip()
        self.df = pd.read_csv(dataset_path, encoding='utf-8')
        
        # Load metadata
        with open(config["dataset_metadata_path"], "r", encoding='utf-8') as f:
            self.dataset_info = json.load(f)
        
        # Conversation history
        self.conversation_history = []
        
        print("Chatbot initialized successfully!")
        print(f"Vector DB: {self.pdf_collection.count()} documents")
        print(f"Dataset: {len(self.df)} rows")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for query using Sentence Transformers."""
        from sentence_transformers import SentenceTransformer
        
        if not hasattr(self, 'st_model'):
            print("Loading embedding model...")
            self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        embedding = self.st_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def classify_and_route_query(self, query: str) -> Dict:
        """
        Advanced query classification and routing with intent detection
        """
        classification_prompt = f"""You are an intelligent query classifier for an e-commerce chatbot system.

Analyze the user's query and classify it into ONE of these categories:

1. **DOCUMENT** - Questions about:
   - Company policies (return, shipping, warranty)
   - Product information from knowledge base
   - General company information
   - How-to guides or documentation

2. **DATASET_ANALYSIS** - Questions requiring data analysis:
   - Product recommendations (best rated, popular, etc.)
   - Statistical queries (average price, top products, etc.)
   - Comparisons between products/categories
   - Finding specific products based on criteria

3. **GENERAL_CHAT** - General conversation:
   - Greetings, small talk
   - Thank you messages
   - Feedback about the service
   - Questions about what the bot can do

For DATASET_ANALYSIS queries, also detect:
- **Output preference**: Does user want SQL/technical output or natural language?
- **Intent**: What specifically is the user asking for?

User Query: "{query}"

Respond in JSON format:
{{
    "category": "DOCUMENT/DATASET_ANALYSIS/GENERAL_CHAT",
    "output_preference": "natural/sql/unclear",
    "intent": "brief description of what user wants",
    "confidence": "high/medium/low"
}}"""

        try:
            response = self.model.generate_content(classification_prompt)
            # Parse JSON from response
            response_text = response.text.strip()
            
            # Extract JSON if wrapped in markdown
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            classification = json.loads(response_text)
            return classification
        except Exception as e:
            print(f"Classification error: {e}")
            # Default classification
            return {
                "category": "DATASET_ANALYSIS",
                "output_preference": "natural",
                "intent": query,
                "confidence": "low"
            }
    
    def query_documents(self, query: str, n_results: int = 3) -> str:
        """Query vector database for relevant documents."""
        query_embedding = self.generate_embedding(query)
        
        results = self.pdf_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        if not results['documents'][0]:
            return "I couldn't find relevant information in the knowledge base. Could you rephrase your question?"
        
        # Combine retrieved context
        context = "\n\n---\n\n".join(results['documents'][0])
        
        # Generate natural language answer
        answer_prompt = f"""You are a helpful customer service assistant for Amazon.

Based on the following information from our knowledge base, answer the customer's question in a friendly and informative way.

Knowledge Base Context:
{context}

Customer Question: {query}

Instructions:
- Provide a clear, concise answer
- Be friendly and professional
- If the information isn't in the context, say so politely
- Use natural language, not robotic responses

Answer:"""

        response = self.model.generate_content(answer_prompt)
        return response.text
    
    def analyze_dataset_natural(self, query: str, classification: Dict) -> str:
        """
        Analyze dataset and provide natural language response
        Uses advanced prompting for better results
        """
        # Generate pandas code to answer the query
        analysis_prompt = f"""You are a data analyst expert. Generate Python pandas code to answer the user's question.

Dataset Schema:
- Columns: {', '.join(self.dataset_info['columns'])}
- Numeric columns: {', '.join(self.dataset_info['numeric_columns'])}
- Sample data: {json.dumps(self.dataset_info['sample_rows'][:2], indent=2)}

Column Descriptions:
{json.dumps(self.dataset_info['column_descriptions'], indent=2)}

User Question: {query}
Intent: {classification['intent']}

Generate pandas code that:
1. Analyzes the data to answer the question
2. Stores the result in a variable called 'result'
3. Is safe to execute (no file operations, imports, etc.)

Use 'df' as the DataFrame variable name.

Return ONLY the Python code, no explanations:"""

        code_response = self.model.generate_content(analysis_prompt)
        pandas_code = code_response.text.strip()
        
        # Clean up code
        if "```python" in pandas_code:
            pandas_code = pandas_code.split("```python")[1].split("```")[0]
        elif "```" in pandas_code:
            pandas_code = pandas_code.split("```")[1].split("```")[0]
        
        pandas_code = pandas_code.strip()
        
        # Execute the code
        result = None
        error = None
        
        try:
            local_vars = {'df': self.df, 'pd': pd}
            exec(pandas_code, {'__builtins__': {}}, local_vars)
            
            if 'result' in local_vars:
                result = local_vars['result']
            else:
                # Find any dataframe/series/value in local_vars
                for var_name, var_value in local_vars.items():
                    if var_name not in ['df', 'pd']:
                        result = var_value
                        break
        except Exception as e:
            error = str(e)
        
        # Generate natural language response
        if error:
            return f"I encountered an error while analyzing the data: {error}\n\nCould you rephrase your question?"
        
        # Convert result to readable format
        result_str = self._format_result(result)
        
        # Generate natural language answer
        nl_prompt = f"""You are a friendly e-commerce assistant helping a customer understand data insights.

Customer Question: {query}

Data Analysis Result:
{result_str}

Generate a natural, conversational response that:
1. Directly answers the customer's question
2. Presents the data insights in an easy-to-understand way
3. Uses bullet points or formatting where helpful
4. Adds helpful context or recommendations if relevant
5. Is enthusiastic and helpful in tone

Response:"""

        nl_response = self.model.generate_content(nl_prompt)
        
        return nl_response.text
    
    def analyze_dataset_sql(self, query: str, classification: Dict) -> str:
        """Analyze dataset and provide SQL + results."""
        # Generate SQL-style query
        sql_prompt = f"""Generate a SQL query to answer this question about an Amazon products dataset.

Dataset Schema:
Columns: {', '.join(self.dataset_info['columns'])}
Column Descriptions: {json.dumps(self.dataset_info['column_descriptions'], indent=2)}

Question: {query}

Provide:
1. A SQL query (assume table name is 'products')
2. Equivalent pandas code

Format:
SQL: <query>
PANDAS: <code>"""

        response = self.model.generate_content(sql_prompt)
        response_text = response.text
        
        # Parse SQL and Pandas code
        sql_query = ""
        pandas_code = ""
        
        for line in response_text.split('\n'):
            if line.startswith('SQL:'):
                sql_query = line.replace('SQL:', '').strip()
            elif line.startswith('PANDAS:'):
                pandas_code = line.replace('PANDAS:', '').strip()
            elif sql_query and not pandas_code and not line.startswith('SQL:'):
                sql_query += " " + line.strip()
            elif pandas_code:
                pandas_code += "\n" + line
        
        # Execute pandas code
        pandas_code = pandas_code.strip()
        if "```" in pandas_code:
            pandas_code = pandas_code.split("```")[1].split("```")[0]
            if pandas_code.startswith('python'):
                pandas_code = '\n'.join(pandas_code.split('\n')[1:])
        
        result = None
        try:
            local_vars = {'df': self.df, 'pd': pd}
            exec(pandas_code, {'__builtins__': {}}, local_vars)
            if 'result' in local_vars:
                result = local_vars['result']
        except Exception as e:
            result = f"Error: {e}"
        
        # Format output
        output = f"**SQL Query:**\n```sql\n{sql_query}\n```\n\n"
        output += f"**Pandas Code:**\n```python\n{pandas_code}\n```\n\n"
        output += f"**Result:**\n{self._format_result(result)}"
        
        return output
    
    def _format_result(self, result) -> str:
        """Format analysis result for display."""
        if result is None:
            return "No result"
        elif isinstance(result, pd.DataFrame):
            if len(result) > 20:
                return result.head(20).to_string() + f"\n\n... ({len(result)} total rows)"
            return result.to_string()
        elif isinstance(result, pd.Series):
            if len(result) > 20:
                return result.head(20).to_string() + f"\n\n... ({len(result)} total items)"
            return result.to_string()
        elif isinstance(result, (list, tuple)):
            return str(result[:20]) + (f"\n... ({len(result)} total items)" if len(result) > 20 else "")
        else:
            return str(result)
    
    def handle_general_chat(self, query: str) -> str:
        """Handle general conversation."""
        chat_prompt = f"""You are a friendly Amazon customer service chatbot assistant.

Respond to this message naturally and helpfully: "{query}"

You can:
- Answer questions about company policies (stored in knowledge base)
- Analyze product data and provide recommendations
- Help users find products
- Provide general assistance

Keep your response friendly, concise, and helpful.

Response:"""

        response = self.model.generate_content(chat_prompt)
        return response.text
    
    def chat(self, message: str, history: List[Tuple[str, str]]) -> str:
        """
        Main chat interface for Gradio
        """
        if not message.strip():
            return "Please enter a message."
        
        # Check for simple direct queries (no API needed)
        response = self._try_direct_query(message)
        if response:
            return response
        
        try:
            # Classify query
            classification = self.classify_and_route_query(message)
            
            print(f"\n Query: {message}")
            print(f"   Category: {classification['category']}")
            print(f"   Output: {classification['output_preference']}")
            print(f"   Intent: {classification['intent']}")
            
            # Route based on classification
            if classification['category'] == 'DOCUMENT':
                response = self.query_documents(message)
                
            elif classification['category'] == 'DATASET_ANALYSIS':
                # Check if user wants SQL output or natural language
                if classification['output_preference'] == 'sql':
                    response = self.analyze_dataset_sql(message, classification)
                else:
                    # Natural language is default
                    response = self.analyze_dataset_natural(message, classification)
                                
            else:  # GENERAL_CHAT
                response = self.handle_general_chat(message)
            
            return response
        
        except Exception as e:
            # If API quota exceeded, try direct query
            if "429" in str(e) or "quota" in str(e).lower():
                return "API quota exceeded. Using direct data access...\n\n" + self._try_direct_query(message, force=True)
            return f"I encountered an error: {str(e)}\n\nPlease try rephrasing your question."
    
    def _try_direct_query(self, message: str, force: bool = False) -> str:
        """
        Handle simple queries directly without API calls
        """
        msg_lower = message.lower()
        
        # Dataset info queries
        if any(word in msg_lower for word in ['how many', 'total rows', 'count', 'dataset size']):
            return f"Dataset Overview:\n- Total products: {len(self.df):,}\n- Total columns: {len(self.df.columns)}\n- Categories: {self.df['main_category'].nunique() if 'main_category' in self.df.columns else 'N/A'}"
        
        # Show columns
        if 'columns' in msg_lower or 'fields' in msg_lower:
            cols = ", ".join(self.df.columns)
            return f"Available Columns:\n{cols}"
        
        # Top rated products
        if 'top rated' in msg_lower or 'best rated' in msg_lower or 'highest rated' in msg_lower:
            top_products = self.df.nlargest(10, 'rating')[['product_name', 'rating', 'rating_count', 'discounted_price']]
            return f"Top 10 Rated Products:\n\n{top_products.to_string(index=False)}"
        
        # Cheapest products
        if 'cheapest' in msg_lower or 'lowest price' in msg_lower:
            cheap = self.df.nsmallest(10, 'discounted_price')[['product_name', 'discounted_price', 'rating']]
            return f"10 Cheapest Products:\n\n{cheap.to_string(index=False)}"
        
        # Most expensive
        if 'expensive' in msg_lower or 'highest price' in msg_lower:
            expensive = self.df.nlargest(10, 'discounted_price')[['product_name', 'discounted_price', 'rating']]
            return f"10 Most Expensive Products:\n\n{expensive.to_string(index=False)}"
        
        # Average price
        if 'average price' in msg_lower or 'mean price' in msg_lower:
            avg = self.df['discounted_price'].mean()
            median = self.df['discounted_price'].median()
            return f"Price Statistics:\n- Average: ₹{avg:.2f}\n- Median: ₹{median:.2f}\n- Min: ₹{self.df['discounted_price'].min():.2f}\n- Max: ₹{self.df['discounted_price'].max():.2f}"
        
        # Categories
        if 'categories' in msg_lower and ('show' in msg_lower or 'list' in msg_lower or 'all' in msg_lower):
            if 'main_category' in self.df.columns:
                cats = self.df['main_category'].value_counts().head(10)
                result = "Top Categories:\n\n"
                for cat, count in cats.items():
                    result += f"- {cat}: {count} products\n"
                return result
        
        # Price range filter
        if 'under' in msg_lower or 'below' in msg_lower:
            import re
            price_match = re.search(r'(\d+)', msg_lower)
            if price_match:
                price = int(price_match.group(1))
                filtered = self.df[self.df['discounted_price'] < price].nlargest(10, 'rating')[['product_name', 'discounted_price', 'rating']]
                return f"Top Products Under ₹{price}:\n\n{filtered.to_string(index=False)}"
        
        # Rating filter
        if 'rating above' in msg_lower or 'rating >' in msg_lower:
            import re
            rating_match = re.search(r'(\d+\.?\d*)', msg_lower)
            if rating_match:
                rating = float(rating_match.group(1))
                filtered = self.df[self.df['rating'] > rating][['product_name', 'rating', 'rating_count', 'discounted_price']].head(10)
                return f"Products with Rating > {rating}:\n\n{filtered.to_string(index=False)}"
        
        # Brands
        if 'brands' in msg_lower or 'brand list' in msg_lower:
            if 'brand' in self.df.columns:
                brands = self.df['brand'].value_counts().head(15)
                result = "Top Brands:\n\n"
                for brand, count in brands.items():
                    result += f"- {brand}: {count} products\n"
                return result
        
        # Average discount
        if 'discount' in msg_lower and ('average' in msg_lower or 'mean' in msg_lower):
            avg_discount = self.df['discount_percentage'].mean()
            return f"Discount Statistics:\n- Average Discount: {avg_discount:.2f}%\n- Max Discount: {self.df['discount_percentage'].max():.2f}%\n- Min Discount: {self.df['discount_percentage'].min():.2f}%"
        
        # If force mode (API failed), try a generic summary
        if force:
            return f"""Dataset Summary:

General Info:
- Total Products: {len(self.df):,}
- Columns: {len(self.df.columns)}

Price Stats:
- Average Price: ₹{self.df['discounted_price'].mean():.2f}
- Price Range: ₹{self.df['discounted_price'].min():.2f} - ₹{self.df['discounted_price'].max():.2f}

Rating Stats:
- Average Rating: {self.df['rating'].mean():.2f}
- Highest Rated: {self.df['rating'].max():.2f}

Top Category:
- {self.df['main_category'].value_counts().index[0] if 'main_category' in self.df.columns else 'N/A'}: {self.df['main_category'].value_counts().iloc[0] if 'main_category' in self.df.columns else 0} products

Try these queries (no API needed):
- "show top rated products"
- "what's the average price"
- "list all categories"
- "show cheapest products"
"""
        
        return None  # No direct match, need API


# Initialize chatbot
GEMINI_API_KEY = ""   # Replace with your API key
chatbot = SmartAmazonChatbot(GEMINI_API_KEY)


# Gradio interface
def chat_interface(message, history):
    """Gradio chat interface wrapper."""
    try:
        response = chatbot.chat(message, history)
        return response
    except Exception as e:
        return f"I encountered an error: {str(e)}\n\nPlease try rephrasing your question."


# Create Gradio interface
with gr.Blocks(
    title="Amazon Smart Assistant",
    theme=gr.themes.Base(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate"
    ),
    css="""
    * {
        font-family: 'Segoe UI', 'Helvetica Neue', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .header-container {
        background: linear-gradient(135deg, #1e3a8a 0%, #1f2937 100%);
        padding: 40px 20px;
        border-radius: 8px;
        margin-bottom: 30px;
    }
    
    .header-title {
        color: white;
        font-size: 32px;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        color: #e5e7eb;
        font-size: 14px;
        margin-top: 8px;
        font-weight: 400;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 20px;
        margin: 30px 0;
    }
    
    
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #1f2937;
        margin-top: 30px;
        margin-bottom: 16px;
        border-left: 3px solid #3b82f6;
        padding-left: 12px;
    }
    
    .tips-container {
        background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 20px;
        margin-top: 30px;
    }
    
    .tips-container ul {
        margin: 0;
        padding-left: 20px;
    }
    
    .tips-container li {
        margin: 8px 0;
        font-size: 13px;
        color: #1e40af;
        line-height: 1.6;
    }
    
    #component-0 {
        max-width: 100%;
    }
    
    .chatbot-interface {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        overflow: hidden;
    }
    """
) as demo:
    # Header Section
    with gr.Group(elem_classes="header-container"):
        gr.HTML("""
            <div class="header-title">Amazon Smart Assistant</div>
            <div class="header-subtitle">Enterprise-Grade AI-Powered E-Commerce Intelligence Platform</div>
        """)

    
    # Chat Interface
    with gr.Group(elem_classes="chatbot-interface"):
        chatbot_ui = gr.ChatInterface(
            fn=chat_interface,
            examples=[
                "What is the return policy?",
                "Suggest me the best rated USB cables under ₹500",
                "Show me the top 5 products by rating",
                "What's the average discount percentage across all products?",
                "Which category has the most products?",
                "Find products with rating above 4.5 - show SQL query",
                "Compare prices between different brands",
                "What are the most popular products?"
            ],
            title="Conversational Interface",
            description="Enter your query below to get started"
        )
    



# Launch the app
if __name__ == "__main__":
    print("\n" + "="*60)
    print("LAUNCHING GRADIO INTERFACE")
    print("="*60)
    print("\nSystem Ready!")
    print(f"Knowledge Base: {chatbot.pdf_collection.count()} documents")
    print(f"Dataset: {len(chatbot.df):,} products")
    print("\nStarting web interface...")
    
    demo.launch(
        share=True,  # Set to True to create a public link
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
