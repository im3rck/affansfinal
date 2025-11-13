"""
Advanced RAG System with Query Rewriting, Hybrid Search, and Reranking
Implements state-of-the-art retrieval techniques for improved chatbot responses
"""

import numpy as np
import google.generativeai as genai
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import re
from collections import defaultdict


class QueryRewriter:
    """
    Advanced Query Rewriting system that expands and reformulates queries
    for better retrieval performance
    """

    def __init__(self, gemini_model):
        self.model = gemini_model

    def rewrite_query(self, original_query: str, context: str = "") -> List[str]:
        """
        Generate multiple query variations using different techniques:
        1. Synonym expansion
        2. Question reformulation
        3. Context-aware expansion
        4. Sub-query decomposition
        """

        rewriting_prompt = f"""You are an expert at query expansion for information retrieval systems.

Original Query: "{original_query}"
{f"Context: {context}" if context else ""}

Generate 3 alternative versions of this query that would help retrieve relevant information:
1. A version with synonyms and related terms
2. A more detailed/specific version
3. A simpler/broader version

Return ONLY the 3 alternative queries, one per line, without numbering or explanations.
Example format:
query variation 1
query variation 2
query variation 3
"""

        try:
            response = self.model.generate_content(rewriting_prompt)
            rewritten_queries = [q.strip() for q in response.text.strip().split('\n') if q.strip()]

            # Always include the original query
            all_queries = [original_query] + rewritten_queries[:3]
            return all_queries

        except Exception as e:
            print(f"Query rewriting error: {e}")
            return [original_query]


class HybridSearcher:
    """
    Implements Hybrid Search combining:
    - Semantic Search (dense embeddings)
    - Keyword Search (BM25 sparse retrieval)
    - Weighted fusion of results
    """

    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.bm25 = None
        self.documents = []
        self.document_embeddings = None

    def index_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """
        Index documents for both semantic and keyword search
        """
        self.documents = documents
        self.metadatas = metadatas if metadatas else [{}] * len(documents)

        # Create BM25 index (keyword search)
        tokenized_docs = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        # Create dense embeddings (semantic search)
        print("Creating document embeddings for hybrid search...")
        self.document_embeddings = self.embedding_model.encode(
            documents,
            convert_to_tensor=False,
            show_progress_bar=True
        )

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\w+', text.lower())
        return tokens

    def search(self, query: str, top_k: int = 10,
               semantic_weight: float = 0.5,
               keyword_weight: float = 0.5) -> List[Dict]:
        """
        Perform hybrid search combining semantic and keyword search

        Args:
            query: Search query
            top_k: Number of results to return
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for keyword search (0-1)
        """

        if not self.documents:
            return []

        # 1. Semantic Search (Dense Retrieval)
        query_embedding = self.embedding_model.encode([query])[0]
        semantic_scores = np.dot(self.document_embeddings, query_embedding)

        # Normalize semantic scores to [0, 1]
        semantic_scores = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-10)

        # 2. Keyword Search (BM25)
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normalize BM25 scores to [0, 1]
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        # 3. Combine scores with weights
        hybrid_scores = (semantic_weight * semantic_scores +
                        keyword_weight * bm25_scores)

        # 4. Get top-k results
        top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'metadata': self.metadatas[idx],
                'score': float(hybrid_scores[idx]),
                'semantic_score': float(semantic_scores[idx]),
                'bm25_score': float(bm25_scores[idx]),
                'index': int(idx)
            })

        return results


class DocumentReranker:
    """
    Reranks retrieved documents using a Cross-Encoder model
    for more accurate relevance scoring
    """

    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        print(f"Loading reranker model: {model_name}...")
        self.cross_encoder = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank documents using cross-encoder for better relevance

        Args:
            query: The search query
            documents: List of document dictionaries with 'document' key
            top_k: Number of top documents to return after reranking
        """

        if not documents:
            return []

        # Prepare query-document pairs
        pairs = [[query, doc['document']] for doc in documents]

        # Get reranking scores
        rerank_scores = self.cross_encoder.predict(pairs)

        # Add rerank scores to documents
        for doc, score in zip(documents, rerank_scores):
            doc['rerank_score'] = float(score)

        # Sort by rerank score
        reranked_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

        return reranked_docs[:top_k]


class AdvancedRAGSystem:
    """
    Complete Advanced RAG System integrating:
    - Query Rewriting
    - Hybrid Search (Semantic + Keyword)
    - Reranking with Cross-Encoder
    """

    def __init__(self, gemini_model, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        self.query_rewriter = QueryRewriter(gemini_model)
        self.hybrid_searcher = HybridSearcher(embedding_model_name)
        self.reranker = DocumentReranker()
        self.indexed = False

    def index_knowledge_base(self, documents: List[str], metadatas: List[Dict] = None):
        """Index documents for retrieval"""
        print("Indexing knowledge base for advanced RAG...")
        self.hybrid_searcher.index_documents(documents, metadatas)
        self.indexed = True
        print(f"✓ Indexed {len(documents)} documents")

    def retrieve(self, query: str,
                use_query_rewriting: bool = True,
                use_reranking: bool = True,
                top_k: int = 5,
                semantic_weight: float = 0.6,
                keyword_weight: float = 0.4) -> Dict[str, Any]:
        """
        Advanced retrieval pipeline with all techniques

        Returns:
            Dictionary containing:
            - original_query: The original query
            - rewritten_queries: List of query variations (if enabled)
            - retrieved_documents: Final top-k documents
            - retrieval_details: Detailed scoring information
        """

        if not self.indexed:
            raise ValueError("Knowledge base not indexed. Call index_knowledge_base() first.")

        results = {
            'original_query': query,
            'rewritten_queries': [],
            'retrieved_documents': [],
            'retrieval_details': {}
        }

        # Step 1: Query Rewriting (if enabled)
        queries_to_search = [query]
        if use_query_rewriting:
            queries_to_search = self.query_rewriter.rewrite_query(query)
            results['rewritten_queries'] = queries_to_search
            print(f"🔄 Query rewriting generated {len(queries_to_search)} queries")

        # Step 2: Hybrid Search for all queries
        all_results = []
        for q in queries_to_search:
            search_results = self.hybrid_searcher.search(
                q,
                top_k=top_k * 2,  # Retrieve more for reranking
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight
            )
            all_results.extend(search_results)

        # Remove duplicates based on document content
        seen_docs = set()
        unique_results = []
        for result in all_results:
            doc_hash = hash(result['document'])
            if doc_hash not in seen_docs:
                seen_docs.add(doc_hash)
                unique_results.append(result)

        print(f"🔍 Hybrid search retrieved {len(unique_results)} unique documents")

        # Step 3: Reranking (if enabled)
        if use_reranking and unique_results:
            final_results = self.reranker.rerank(query, unique_results, top_k=top_k)
            print(f"📊 Reranked to top {len(final_results)} documents")
        else:
            final_results = sorted(unique_results, key=lambda x: x['score'], reverse=True)[:top_k]

        results['retrieved_documents'] = final_results
        results['retrieval_details'] = {
            'total_retrieved': len(unique_results),
            'final_count': len(final_results),
            'used_query_rewriting': use_query_rewriting,
            'used_reranking': use_reranking
        }

        return results

    def get_context_for_llm(self, retrieval_results: Dict) -> str:
        """
        Format retrieved documents as context for LLM
        """
        documents = retrieval_results['retrieved_documents']

        if not documents:
            return ""

        context_parts = []
        for i, doc in enumerate(documents, 1):
            score_info = f"[Relevance: {doc.get('rerank_score', doc.get('score', 0)):.3f}]"
            context_parts.append(f"Document {i} {score_info}:\n{doc['document']}")

        return "\n\n---\n\n".join(context_parts)


# Utility function for easy integration
def create_advanced_rag_system(gemini_model) -> AdvancedRAGSystem:
    """Factory function to create and initialize the advanced RAG system"""
    return AdvancedRAGSystem(gemini_model)
