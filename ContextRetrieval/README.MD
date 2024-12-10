![Contextual vs Regular Rag]("./context_vs_regular_rag.png")

# Overview

This code implements Contextual RAG System that combines vector-based similarity search with keyword-based BM25 retrieval. The approach aims to leverage the strengths of both methods to improve the overall quality and relevance of document retrieval.

# Motivation

Traditional retrieval methods often rely on either semantic understanding (vector-based) or keyword matching (BM25). Each approach has its strengths and weaknesses. Fusion retrieval aims to combine these methods to create a more robust and accurate retrieval system that can handle a wider range of queries effectively. The aim of this notebook is to compare Contextual Retrieval implementation with "traditional" implementation.

# Key Components

1. "m-ric/huggingface_doc_qa_eval" Hugging Face dataset
2. Pinecone Vector store for embedding storage
3. OpenAI embeddings
4. OpenAI summary model and generation model (Can be any other model)
5. BM25 index creation for keyword-based retrieval
6. Custom fusion retrieval function that combines both methods

# Method Details

## Document Preprocessing
1. The dataset "m-ric/huggingface_doc_qa_eval" is loaded and filtered to keep only high-quality question/answer pairs (standalone_score >= 4)
2. Documents are split into chunks using RecursiveCharacterTextSplitter with:
   - Chunk size: 800 characters
   - Overlap: 200 characters
   - Custom markdown separators to maintain document structure

## Document Contextualization
1. Each chunk is enriched with contextual information using OpenAI GPT model:
   - A prompt template guides the model to analyze how each chunk relates to its parent document
   - Generated context is concise (3-4 sentences) and captures the chunk's role within the broader document
   - The context is prepended to the chunk text for enhanced retrieval

## Vector Store Creation
1. OpenAI embeddings (text-embedding-3-small model) are used to create vector representations of:
   - Regular chunks (without context)
   - Contextualized chunks (with prepended context)
2. Two separate Pinecone vector stores (ServerlessSpec) are created:
   - One for regular chunks
   - One for contextualized chunks
   - Both use cosine similarity metric and 1536 dimensions

## BM25 Index Creation
1. Two BM25Okapi indexes are created using NLTK word tokenization:
   - One for regular chunks
   - One for contextualized chunks
2. This enables keyword-based retrieval alongside vector-based methods

## Fusion Retrieval Function
The fusion_rank_search function combines multiple retrieval approaches:

1. Initial Retrieval:
   - Performs both vector-based (Pinecone) and BM25-based retrieval
   - Gets top-k (default 20) results from each method
   - Normalizes scores from both methods to a common scale (0-1)

2. Score Combination:
   - Weighted combination using the weight_sparse (alpha) parameter
   - Aggregates scores for documents appearing in both result sets
   - Normalizes combined scores by the number of methods that retrieved each document

3. Reranking:
   - Uses BAAI/bge-reranker-v2-m3 model to rerank the combined results
   - Query-document pairs are scored by the reranker
   - Final ranking is based on reranker scores

4. Returns the top-k (default 5) documents after reranking

## Evaluation
Using BERTScore metrics to compare the effectiveness of regular vs. contextualized retrieval approaches.

# Benefits of This Approach
1. Improved Retrieval Quality: By combining semantic and keyword-based search, the system can capture both conceptual similarity and exact keyword matches.
2. Flexibility: The alpha parameter allows for adjusting the balance between vector and keyword search based on specific use cases or query types.
3. Robustness: The combined approach can handle a wider range of queries effectively, mitigating weaknesses of individual methods.
4. Customizability: The system can be easily adapted to use different vector stores or keyword-based retrieval methods.

# Conclusion
Fusion retrieval represents a powerful approach to document search that combines the strengths of semantic understanding and keyword matching. By leveraging both vector-based and BM25 retrieval methods, it offers a more comprehensive and flexible solution for information retrieval tasks. This approach has potential applications in various fields where both conceptual similarity and keyword relevance are important, such as academic research, legal document search, or general-purpose search engines.

Averaged results show slightly better performance of contextual retrieval vs. regular. There are several parameters that can be played with (chunking size, chunk overlap, alpha for fusion score calculations) and have impact on final result.