# Advanced RAG Chatbot with Hybrid ML + Gen AI Analytics

An enterprise-grade AI-powered e-commerce intelligence platform combining advanced RAG techniques with hybrid ML and generative AI for comprehensive data analysis and visualization.

## 🚀 Features

### Advanced RAG System
This implementation includes **three advanced RAG techniques**:

1. **Query Rewriting**
   - Automatically generates multiple query variations
   - Expands queries with synonyms and related terms
   - Creates broader and more specific versions for comprehensive retrieval
   - Improves recall by capturing different ways users might phrase questions

2. **Hybrid Search** (Semantic + Keyword)
   - Combines dense retrieval (semantic search via embeddings)
   - Integrates sparse retrieval (BM25 keyword matching)
   - Weighted fusion of both approaches (configurable weights)
   - Captures both semantic meaning and exact keyword matches

3. **Reranking**
   - Uses Cross-Encoder model for accurate relevance scoring
   - Reranks retrieved documents for optimal results
   - Significantly improves retrieval precision
   - Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`

### Hybrid ML + Gen AI Analytics

The system combines traditional machine learning with generative AI:

#### Traditional ML Analysis
- **Price Analysis**: Statistical analysis, distribution, trends
- **Sales Trends**: Category-wise performance, demand patterns
- **Customer Behavior**: Rating patterns, preferences, correlations
- **Clustering**: K-Means clustering for product segmentation
- **Statistical Methods**: Correlation analysis, trend detection

#### Automated Chart Generation
Creates insightful visualizations using Matplotlib and Plotly:
- **Price Distribution Charts**: Histograms, box plots, statistics
- **Category Analysis**: Bar charts, performance comparisons
- **Rating Analysis**: Scatter plots, distributions, correlations
- **Discount Patterns**: Trend analysis, pricing strategies
- **Clustering Visualizations**: Product segment identification

#### LLM-Powered Interpretation
- Gemini AI interprets ML results and charts
- Generates natural language insights
- Provides actionable business recommendations
- Combines quantitative analysis with qualitative interpretation

## 📁 Project Structure

```
affansfinal/
├── frontend.py              # Main application with Gradio UI
├── advanced_rag.py          # Advanced RAG implementation
│   ├── QueryRewriter        # Query expansion & reformulation
│   ├── HybridSearcher       # Semantic + BM25 search
│   ├── DocumentReranker     # Cross-encoder reranking
│   └── AdvancedRAGSystem    # Complete RAG pipeline
├── data_analytics.py        # Hybrid ML + Gen AI analytics
│   ├── DataAnalyzer         # Traditional ML analysis
│   ├── ChartGenerator       # Matplotlib/Plotly visualizations
│   └── HybridAnalyticsEngine # ML + LLM integration
├── requirements.txt         # Python dependencies
├── config.json             # Configuration settings
├── dataset_metadata.json   # Dataset schema information
└── chroma_db/              # Vector database storage
```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd affansfinal
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up API key**
Edit `frontend.py` and add your Gemini API key:
```python
GEMINI_API_KEY = "your-api-key-here"
```

## 🚀 Usage

### Running the Chatbot

```bash
python frontend.py
```

The application will:
1. Initialize the advanced RAG system
2. Load and index documents from ChromaDB
3. Set up the hybrid analytics engine
4. Launch the Gradio web interface on `http://0.0.0.0:7860`

### Example Queries

#### Document Queries (Advanced RAG)
- "What is the return policy?"
- "Tell me about warranty information"
- "How does shipping work?"

These queries use:
- Query rewriting for better coverage
- Hybrid search (semantic + keyword)
- Cross-encoder reranking

#### Data Analysis Queries
- "Suggest the best rated USB cables under ₹500"
- "What's the average discount percentage?"
- "Find products with rating above 4.5"

#### Analytics & Visualization Queries
- **"Show me price distribution analysis with charts"**
- **"Visualize rating trends and patterns"**
- **"Show sales insights by category with graphs"**
- **"Analyze discount patterns and create visualizations"**
- **"What are the pricing trends across categories?"**
- **"Show comprehensive data analysis with charts"**

These queries trigger:
- ML-based statistical analysis
- Automatic chart generation
- LLM interpretation of insights

## 📊 System Architecture

### Data Flow

```
User Query
    ↓
Query Classification
    ↓
├─→ DOCUMENT           → Advanced RAG Pipeline
│                        ├─ Query Rewriting
│                        ├─ Hybrid Search (Semantic + BM25)
│                        ├─ Cross-Encoder Reranking
│                        └─ LLM Answer Generation
│
├─→ ANALYTICS          → Hybrid ML + Gen AI
│                        ├─ Traditional ML Analysis
│                        ├─ Chart Generation (Matplotlib/Plotly)
│                        └─ LLM Interpretation
│
└─→ DATASET_ANALYSIS   → Direct SQL/Pandas Query
                         └─ Natural Language Response
```

## 🔧 Technical Implementation

### Advanced RAG Pipeline

```python
# Query Rewriting
original_query = "return policy"
rewritten = ["return and refund policy", "product return guidelines", "refund process"]

# Hybrid Search
semantic_results = embedding_search(query)  # Dense retrieval
keyword_results = bm25_search(query)        # Sparse retrieval
hybrid_results = weighted_fusion(semantic_results, keyword_results)

# Reranking
final_results = cross_encoder_rerank(query, hybrid_results)
```

### Hybrid Analytics

```python
# ML Analysis
ml_insights = {
    'price_stats': statistical_analysis(df),
    'trends': trend_detection(df),
    'clusters': kmeans_clustering(df)
}

# Chart Generation
charts = [
    generate_price_distribution(),
    generate_category_analysis(),
    generate_rating_patterns()
]

# LLM Interpretation
interpretation = gemini.interpret(ml_insights, charts)
```

## 📈 Performance Characteristics

### Advanced RAG Benefits
- **Improved Recall**: Query rewriting captures diverse phrasings
- **Better Precision**: Hybrid search combines semantic + keyword matching
- **Higher Accuracy**: Reranking ensures most relevant documents surface first
- **Robust Retrieval**: Multiple techniques provide fallback mechanisms

### Analytics Benefits
- **Data-Driven Insights**: ML analysis uncovers patterns
- **Visual Understanding**: Charts make complex data accessible
- **Business Intelligence**: LLM provides actionable recommendations
- **Automated Analysis**: Reduces manual data exploration time

## 🔒 Security & Best Practices

- API keys should be stored in environment variables (not hardcoded)
- Input validation on all user queries
- Safe code execution for pandas operations
- Rate limiting on API calls
- Secure data handling practices

## 📝 Configuration

### Advanced RAG Settings
```python
# In advanced_rag.py
semantic_weight = 0.6  # Weight for semantic search
keyword_weight = 0.4   # Weight for BM25
top_k = 5             # Number of documents to retrieve
use_query_rewriting = True
use_reranking = True
```

### Analytics Settings
```python
# In data_analytics.py
n_clusters = 4        # K-means clustering
chart_dpi = 100      # Chart resolution
sample_size = 1000   # For scatter plots
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Gemini AI**: LLM for query understanding and interpretation
- **Sentence Transformers**: Semantic embeddings
- **ChromaDB**: Vector database
- **rank-bm25**: Keyword search implementation
- **scikit-learn**: ML algorithms
- **Matplotlib/Plotly**: Data visualization
- **Gradio**: Web interface

## 📞 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Built with ❤️ using Advanced RAG + Hybrid ML + Gen AI**
