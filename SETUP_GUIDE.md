# Setup Guide - Advanced RAG Chatbot

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- **Core**: gradio, pandas, numpy
- **AI/ML**: google-generativeai, sentence-transformers, chromadb
- **Advanced RAG**: rank-bm25, transformers
- **Visualization**: matplotlib, seaborn, plotly
- **ML**: scikit-learn, scipy

### 2. Configure API Key

Edit `frontend.py` line 520 and add your Gemini API key:

```python
GEMINI_API_KEY = "your-gemini-api-key-here"
```

Get your API key from: https://makersuite.google.com/app/apikey

### 3. Run the Application

```bash
python frontend.py
```

The application will:
- Initialize Advanced RAG System (with Query Rewriting, Hybrid Search, Reranking)
- Load the dataset and ChromaDB knowledge base
- Set up Hybrid Analytics Engine
- Launch web interface at `http://0.0.0.0:7860`

### 4. Access the Interface

Open your browser and go to:
- Local: `http://localhost:7860`
- Network: `http://0.0.0.0:7860`
- Public: Gradio will generate a share link if enabled

## 🎯 What's New?

### Advanced RAG Features

#### 1. Query Rewriting ✨
Automatically expands your query for better results:
```
Original: "return policy"
Expanded:
  - "return and refund policy"
  - "product return guidelines"
  - "refund process"
```

#### 2. Hybrid Search 🔍
Combines two powerful search methods:
- **Semantic Search**: Understands meaning (embeddings)
- **BM25 Keyword Search**: Finds exact matches
- **Weighted Fusion**: Best of both worlds (60% semantic, 40% keyword)

#### 3. Reranking 📊
Uses Cross-Encoder to:
- Score query-document pairs precisely
- Reorder results for maximum relevance
- Ensure best matches appear first

### Hybrid ML + Gen AI Analytics

#### Traditional ML Analysis
- Price distribution and statistics
- Sales trends by category
- Customer behavior patterns
- K-Means product clustering
- Correlation analysis

#### Automated Visualizations
- **Price Charts**: Distribution histograms, box plots
- **Category Analysis**: Bar charts, performance comparison
- **Rating Analysis**: Scatter plots, distributions
- **Discount Patterns**: Trend analysis
- **Clustering**: Product segmentation visualization

#### LLM Interpretation
- Natural language insights
- Business recommendations
- Pattern explanation
- Actionable strategies

## 💬 Example Queries

### Test Advanced RAG (Document Queries)
```
"What is the return policy?"
"Tell me about warranty information"
"How does shipping work?"
"What are the company policies?"
```

**What happens behind the scenes:**
1. Query is rewritten into 3 variations
2. Hybrid search retrieves top documents (semantic + keyword)
3. Cross-encoder reranks for relevance
4. LLM generates natural answer from best documents

### Test Hybrid Analytics (Visualization Queries)
```
"Show me price distribution analysis with charts"
"Visualize rating trends and patterns"
"Show sales insights by category with graphs"
"Analyze discount patterns and create visualizations"
"What are the pricing trends across categories?"
```

**What happens behind the scenes:**
1. ML algorithms analyze data (statistics, clustering, trends)
2. Charts are automatically generated (matplotlib/plotly)
3. LLM interprets results and provides insights
4. Response includes charts embedded as images

### Test Data Analysis (Product Queries)
```
"Suggest the best rated USB cables under ₹500"
"What's the average discount percentage?"
"Find products with rating above 4.5"
"Which category has the most products?"
```

## 🔍 Verification Checklist

After installation, verify these features work:

- [ ] Application starts without errors
- [ ] Advanced RAG initializes (see console output)
- [ ] Analytics Engine initializes (see console output)
- [ ] Document queries return relevant answers
- [ ] Analytics queries generate charts
- [ ] Charts display correctly in the interface
- [ ] All example queries work

## 🐛 Troubleshooting

### Issue: "No module named 'X'"
**Solution**: Install all dependencies
```bash
pip install -r requirements.txt
```

### Issue: "API quota exceeded"
**Solution**:
1. Check your Gemini API key
2. Wait a few minutes (rate limiting)
3. Consider upgrading your API quota

### Issue: ChromaDB errors
**Solution**: Make sure `chroma_db` folder exists and contains data
```bash
ls chroma_db/
```

### Issue: Charts not displaying
**Solution**:
1. Ensure matplotlib, seaborn, plotly are installed
2. Check browser console for errors
3. Try a different browser

### Issue: Slow initialization
**Solution**: First-time setup downloads ML models:
- sentence-transformers model (~80MB)
- cross-encoder model (~90MB)
- This is normal and only happens once

## 📊 Performance Tips

1. **For faster responses**: Reduce `top_k` in advanced_rag.py (default: 5)
2. **For more accurate results**: Increase `top_k` (try 7-10)
3. **For lightweight mode**: Disable reranking by setting `use_reranking=False`
4. **For faster charts**: Reduce sample size in data_analytics.py (default: 1000)

## 🔧 Configuration Options

### Advanced RAG Settings
Edit `frontend.py` in the `query_documents` method:

```python
retrieval_results = self.advanced_rag.retrieve(
    query,
    use_query_rewriting=True,    # Enable/disable query rewriting
    use_reranking=True,           # Enable/disable reranking
    top_k=5,                      # Number of documents to retrieve
    semantic_weight=0.6,          # Weight for semantic search (0-1)
    keyword_weight=0.4            # Weight for BM25 search (0-1)
)
```

### Analytics Settings
Edit `data_analytics.py`:

```python
# Clustering
n_clusters = 4  # Number of product clusters

# Chart resolution
plt.rcParams['figure.figsize'] = (12, 6)  # Chart size
dpi=100  # Resolution (higher = better quality but slower)

# Sample size for scatter plots
sample_size = 1000  # Reduce for faster rendering
```

## 🎓 Understanding the System

### Query Classification Flow
```
User Query
    ↓
LLM Classification
    ↓
┌─────────────────┬──────────────────┬─────────────────┐
│   DOCUMENT      │    ANALYTICS     │ DATASET_ANALYSIS│
│  (Advanced RAG) │ (ML + Charts)    │ (Direct Query)  │
└─────────────────┴──────────────────┴─────────────────┘
```

### Advanced RAG Pipeline
```
Query Input
    ↓
Query Rewriting (3 variations)
    ↓
Hybrid Search (Semantic + BM25)
    ↓
Remove Duplicates
    ↓
Cross-Encoder Reranking
    ↓
Top-K Documents
    ↓
LLM Answer Generation
```

### Analytics Pipeline
```
Query Input
    ↓
ML Analysis (Statistics, Clustering, Trends)
    ↓
Chart Generation (Matplotlib/Plotly)
    ↓
LLM Interpretation
    ↓
Combined Response (Text + Charts)
```

## 📈 Expected Behavior

### First Run
- Takes 2-5 minutes to download ML models
- Indexes documents for hybrid search
- Initializes all systems

### Subsequent Runs
- Starts in 10-30 seconds
- Uses cached models
- Ready immediately

### Query Response Times
- Document queries: 2-5 seconds
- Analytics with charts: 5-10 seconds
- Simple data queries: 1-2 seconds

## 🎉 Success Indicators

When everything is working, you'll see:

```
🚀 LAUNCHING ADVANCED RAG CHATBOT WITH HYBRID ML+AI ANALYTICS
======================================================================

✅ System Ready!
📚 Knowledge Base: X documents
📊 Dataset: X,XXX products

🎯 Advanced Features:
   ✓ Query Rewriting for better retrieval
   ✓ Hybrid Search (Semantic + Keyword/BM25)
   ✓ Cross-Encoder Reranking
   ✓ ML-based Data Analysis
   ✓ Auto-generated Charts & Visualizations
   ✓ LLM-powered Insight Interpretation

🌐 Starting web interface...
```

## 🆘 Need Help?

1. Check this guide first
2. Review README.md for feature documentation
3. Check error messages in console
4. Verify all dependencies are installed
5. Ensure API key is configured correctly

---

**Ready to go? Run `python frontend.py` and start chatting!** 🚀
