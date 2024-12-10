# -*- coding: utf-8 -*-
"""contextual_retrieval.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1U38chUutZASPlWqQxtkOXP4GvQXibgt5

# Overview

This code implements Contextual RAG System that combines vector-based similarity search with keyword-based BM25 retrieval. The approach aims to leverage the strengths of both methods to improve the overall quality and relevance of document retrieval.

#Motivation
Traditional retrieval methods often rely on either semantic understanding (vector-based) or keyword matching (BM25). Each approach has its strengths and weaknesses. Fusion retrieval aims to combine these methods to create a more robust and accurate retrieval system that can handle a wider range of queries effectively. The aim of this notbook to compare Contextual Retrieval implementation with "simple/traditional" implemintation

# Key Components
 - "m-ric/huggingface_doc_qa_eval" Hugging face dataset
 - Pinecone Vector store for embedding storage
 - OpenAI embeddings
 - OpenAI summary model and generation model (Can be any other model)
 - BM25 index creation for keyword-based retrieval
 - Custom fusion retrieval function that combines both methods

# Method Details
Based on the code in the notebook/file, I can enhance the Method Details section to more accurately reflect the implementation:

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

# Benefits of this Approach
1. Improved Retrieval Quality: By combining semantic and keyword-based search, the system can capture both conceptual similarity and exact keyword matches.
2. Flexibility: The alpha parameter allows for adjusting the balance between vector and keyword search based on specific use cases or query types.
3. Robustness: The combined approach can handle a wider range of queries effectively, mitigating weaknesses of individual methods.
4. Customizability: The system can be easily adapted to use different vector stores or keyword-based retrieval methods.

# Conclusion
Fusion retrieval represents a powerful approach to document search that combines the strengths of semantic understanding and keyword matching. By leveraging both vector-based and BM25 retrieval methods, it offers a more comprehensive and flexible solution for information retrieval tasks. This approach has potential applications in various fields where both conceptual similarity and keyword relevance are important, such as academic research, legal document search, or general-purpose search engines.
Averaged results show slightly better performance contextual retrivale vs. regular. There are several parameters that can be played with (chunking size, chunk ovelap, alpha for fusion score calculations) and have impact on final result.
"""

# !pip install sentence_transformers -qU
!pip install rank_bm25 -qU
!pip install datasets -qU
!pip install pinecone[grpc] -qU
!pip install langchain_core -qU
!pip install langchain -qU
!pip install langchain_groq -qU
!pip install langchain-google-genai -qU
!pip install langchain-openai  -qU
# ==0.2.9
!pip install bert-score  -qU

"""# Importing libraries"""

import numpy as np
import nltk
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import pinecone
import pandas as pd # for dataframe
import getpass
from google.colab import userdata
import os

nltk.download('punkt_tab')

"""# Loading dataset"""

# Load dataset from Hugging Face
dataset = load_dataset("m-ric/huggingface_doc_qa_eval")

df = pd.DataFrame(dataset['train'])
print(df.head())

"""## **Taking only best question/answer pairs**"""

best_answers_df = df[df['standalone_score'] >= 4]
print(best_answers_df.head())

best_answers_df.info()

"""# **Logging into Huggng Face**"""

from datasets import Dataset
from huggingface_hub import login


hf_token = userdata.get("HuggingFace")
if not hf_token:
  # Login to Hugging Face (you'll need your token)
  hf_token = input("Please enter your Hugging Face token: ")
login(hf_token)

"""# **Saving best_answers_df to Hugging face to prevent change**"""

best_answers_ds = Dataset.from_pandas(best_answers_df)
# Push to Hugging Face Hub
best_answers_ds.push_to_hub(
    "AIEnthusiast369/hf_doc_qa_eval_best_answers",
    private=False
)

"""# Extract contexts from the dataset"""

texts = best_answers_df['context'].tolist()

"""# **Setting up Embedding model**

## **sentence-transformers**
"""

# # load ' sentence-transformers/all-MiniLM-L6-v2' embedding model from Hugging Face
# from transformers import AutoTokenizer, AutoModel
# model_name = 'sentence-transformers/all-MiniLM-L6-v2'
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# max_seq_length = tokenizer.model_max_length
# embedding_model = AutoModel.from_pretrained(model_name)

"""## **openai**"""

openai_api_key = userdata.get("OPENAI_API_KEY")
if not openai_api_key:
  openai_api_key = getpass("Please enter your OPENAI API KEY: ")

os.environ["OPENAI_API_KEY"] = openai_api_key

from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

max_seq_length = embedding_model.embedding_ctx_length
# index_dimensions = embedding_model.dimensions
index_dimensions = 1536 # default setting of text-embedding-3-small
print(f'max_seq_length:{max_seq_length}, index_dimensions:{index_dimensions}')

"""# Defining text splitter

###openai
"""

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]
# Use RecursiveCharacterTextSplitter to split documents into chunks
chunk_overlap = 200
chunk_size = 800
print('chunk_size',chunk_size)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=MARKDOWN_SEPARATORS,
)

"""# **Definining ProcessedDocument & Chunk**"""

class Chunk:
    def __init__(self, text: str):
        self.text = text
        self.context = None

class ProcessedDocument:
    def __init__(self, text: str, chunks: list[Chunk]):
        self.text = text
        self.chunks = chunks

docs_processed: list[ProcessedDocument] = []
for text in texts:
    # text = doc.page_content  # Extract the text content from the Document
    chunks = text_splitter.split_text(text)  # Split the text into chunks (strings)
    print(f"Number of chunks for document #{len(docs_processed)}: {len(chunks)}")
    processed_doc = ProcessedDocument(
        text,
        [Chunk(chunk_text) for chunk_text in chunks]
    )
    docs_processed.append(processed_doc)
print(f"Number of Processed document: {len(docs_processed)}")

# Count total chunks
total_chunks = sum(len(doc.chunks) for doc in docs_processed)
print(f"Total number of chunks across all documents: {total_chunks}")

"""# **Define summary chain**"""

from langchain.prompts import PromptTemplate
from google.colab import userdata

"""### **OpenAI**"""

from langchain_openai import ChatOpenAI


model_chat_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model=model_chat_name)
sum_provider = 'OPENAI'

prompt_template = ChatPromptTemplate.from_messages([
    ("system",
            """You are an AI assistant specializing in document summarization and contextualization. Your task is to provide brief, relevant context for a specific chunk of text based on a larger document. Here's how to proceed:
"""),
    ("human", """
First, carefully read and analyze the following document:

<document>
{document}
</document>

Now, consider this specific chunk of text from the document:

<chunk>
{chunk}
</chunk>

Your goal is to provide a concise context for this chunk, situating it within the whole document. Follow these guidelines:

1. Analyze how the chunk relates to the overall document's themes, arguments, or narrative.
2. Identify the chunk's role or significance within the broader context of the document.
3. Determine what information from the rest of the document is most relevant to understanding this chunk.

Compose your response as follows:
- Provide 3-4 sentences maximum of context.
- Begin directly with the context, without any introductory phrases.
- Use language like "Focuses on..." or "Addresses..." to describe the chunk's content.
- Ensure the context would be helpful for improving search retrieval of the chunk.

Important notes:
- Do not use phrases like "this chunk" or "this section" in your response.
- Do not repeat the chunk's content verbatim; provide context from the rest of the document.
- Avoid unnecessary details; be succinct and relevant.
- Do not include any additional commentary or meta-discussion about the task itself.

 Remember, your goal is to provide clear, concise, and relevant context that situates the given chunk within the larger document.
            """
     )
])

def create_context_chain(llm):
    return prompt_template | llm

context_chain = create_context_chain(llm)

def get_context(text: str, chunk: str) -> str:
    if len(chunk.strip()) <= 0 or len(text.strip()) <= 0:
        print(f"Chunk or text is empty")
        raise Exception("Chunk or text is empty")
    context= context_chain.invoke({"document": text, "chunk": chunk})
    return context.content

def generate_context(docs_processed: list[ProcessedDocument]):
    for i, doc in enumerate(docs_processed):
        print(f'processing document index {i}')
        for chunk in doc.chunks:
            # print(chunk.text)
            context: str = get_context(text= doc.text, chunk= chunk.text)
            chunk.context = context
            # print(f"chunk with context: Context: \n\n {chunk.context} \n\n Chunk: {chunk.text}")

"""# **Testing chain**"""

page = """
 Convert weights to safetensors

PyTorch model weights are commonly saved and stored as `.bin` files with Python's [`pickle`](https://docs.python.org/3/library/pickle.html) utility. To save and store your model weights in the more secure `safetensor` format, we recommend converting your weights to `.safetensors`.
The easiest way to convert your model weights is to use the [Convert Space](https://huggingface.co/spaces/diffusers/convert), given your model weights are already stored on the Hub. The Convert Space downloads the pickled weights, converts them, and opens a Pull Request to upload the newly converted `.safetensors` file to your repository.
<Tip warning={true}>
For larger models, the Space may be a bit slower because its resources are tied up in converting other models. You can also try running the [convert.py](https://github.com/huggingface/safetensors/blob/main/bindings/python/convert.py) script (this is what the Space is running) locally to convert your weights.
Feel free to ping [@Narsil](https://huggingface.co/Narsil) for any issues with the Space.
</Tip>
"""
chunk = """
Convert weights to safetensors
PyTorch model weights are commonly saved and stored as `.bin` files with Python's [`pickle`](https://docs.python.org/3/library/pickle.html) utility. To save and store your model weights in the more secure `safetensor` format, we recommend converting your weights to `.safetensors`.
The easiest way to convert your model weights is to use the [Convert Space](https://huggingface.co/spaces/diffusers/convert), given your model weights are already stored on the Hub. The Convert Space downloads the pickled weights, converts them, and opens a Pull Request to upload the newly converted `.safetensors` file to your repository.
<Tip warning={true}>
For larger models, the Space may be a bit slower because its resources are tied up in converting other models. You can also try running the [convert.py](https://github.com/huggingface/safetensors/blob/main/bindings/python/convert.py) script (this is what the Space is running) locally to convert your weights.
Feel free to ping [@Narsil](https://huggingface.co/Narsil) for any issues with the Space.
</Tip>
"""

test_context = get_context(text = page, chunk=chunk)

print(test_context)

# temp_docs = docs_processed[1:2]
# generate_context(temp_docs)
generate_context(docs_processed)

"""## Save processed documents to file"""

import joblib
from datetime import datetime
from google.colab import files
import glob
import os

def save_download_object(object, filename):
    joblib.dump(object, filename)
    print(f"Saved object to {filename}")
    files.download(filename)
    print(f"Downloaded {filename}")

def create_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_filename_timestamp(filename, extension = "joblib") -> str:
    timestamp = create_timestamp()
    return f"{filename}_{timestamp}.{extension}"

chunk_texts = []
document_texts = []
contexts = []

# Extract data from docs_processed
for doc in docs_processed:
    for chunk in doc.chunks:
        chunk_texts.append(chunk.text)
        contexts.append(chunk.context)
        document_texts.append(doc.text)

# Create dictionary for dataset
dataset_dict = {
    'chunk': chunk_texts,
    'document': document_texts,
    'context': contexts
}

"""# **Saving Context + Chunks to dataset**"""

# Convert to Hugging Face Dataset
dataset = Dataset.from_dict(dataset_dict)

# Push to Hugging Face Hub
dataset.push_to_hub(
    f"AIEnthusiast369/hf_doc_qa_eval_chunk_size_{chunk_size}_open_ai",  # Replace with your username and desired dataset name
    private=False  # Set to False if you want it public
)

"""# **Loading chunks with context dataset**
*Yuu need to run it only in case of notebook timing out and you loose state*
"""

chunked_dataset = load_dataset("AIEnthusiast369/hf_doc_qa_eval_chunk_size_800_open_ai")
chunks_from_ds=True

if chunks_from_ds:
   best_answers_ds = load_dataset("AIEnthusiast369/hf_doc_qa_eval_best_answers", split="train")
   best_answers_df = best_answers_ds.to_pandas()

"""# **Creating contextualized chunks**"""

chunks_with_context = []
chunks_regular=[]

if chunks_from_ds:
  chuncked_ds = chunked_dataset['train']
  for i in range(len(chuncked_ds)):
      row = chuncked_ds[i]
      chunk = row['chunk']
      chunks_regular.append(chunk)
      context = row['context']
      if context:
              chunks_with_context.append(
                f"{context} \n\n {chunk}"
              )
else:
  for doc in docs_processed:
      for chunk in doc.chunks:
          chunks_regular.append(chunk.text)
          if chunk.context:  # Only include chunks that have a context
              chunks_with_context.append(
                f"{chunk.context} \n\n {chunk.text}"
              )
print(f'Len of regular chunks: {len(chunks_regular)}')
print(f'Len of chunks with context: {len(chunks_with_context)}')

"""# **Setting up Indeses**"""

def create_bm25(chunks: list[str]):
    print("Creating BM25 model...")
    tokenized_chunks = [nltk.word_tokenize(chunk) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    return bm25

from pinecone import Pinecone, ServerlessSpec

pinecone_api_key = userdata.get("PINECONE_API_KEY")
if not pinecone_api_key:
  pinecone_api_key = input("Please enter your PINECONE API KEY: ")

spec=ServerlessSpec(
    cloud="aws",
    region="us-east-1"
  )

EMBEDDING_INDEX_CONTEXTUAL: str = "test-rag-openai-contextual"
EMBEDDING_INDEX_REGULAR: str = "test-rag-openai-regular"

pc = Pinecone(api_key=pinecone_api_key)

from typing import Any, List
from time import sleep

def wait_for_index(index_name):
    while True:
        desc = pc.describe_index(index_name)
        if desc['ready']:
            print("Index is ready!")
            break
        sleep(5)

def create_pinecone_indexes(pinecone, embedding_model, index_name: str, chunks: list[str], specs: ServerlessSpec, dimensions, index_names: List[str]) -> Any:

    if index_name not in index_names:
        pc.create_index(index_name, dimension=dimensions, metric="cosine", spec=specs)
        wait_for_index(index_name)

    # Connect to Pinecone indexes
    embedding_index = pc.Index(index_name)


    # Semantic Embeddings using a Pre-trained Transformer Model
    embeddings = embedding_model.embed_documents(chunks)
    # Store embeddings in Pinecone
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        embedding_index.upsert([(str(i), embedding, {"text": chunk})])

    print(f'len(embeddings)={len(embeddings)}, len(embeddings[0])={len(embeddings[0])}')
    return embedding_index

"""# **Creating Indeses**"""

if not pc.has_index(EMBEDDING_INDEX_CONTEXTUAL):
   create_pinecone_indexes(pc, embedding_model, EMBEDDING_INDEX_CONTEXTUAL, chunks_with_context, spec, 1536, index_names)
if not pc.has_index(EMBEDDING_INDEX_REGULAR):
   create_pinecone_indexes(pc, embedding_model, EMBEDDING_INDEX_REGULAR, chunks_regular, spec, 1536, index_names)
bm25_regular = create_bm25(chunks_regular)
bm25_contextual = create_bm25(chunks_with_context)

"""# **Definining Reranker**

### **Hugging Face**
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RERANKER_MODEL = 'BAAI/bge-reranker-v2-m3'
tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
model.eval()

def get_reranker_score(pairs):
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        print(f'reranker scores {scores}')
        return scores

"""# **Fusion Rank Search**"""

from collections import defaultdict
def fusion_rank_search(
    query: str,
    bm25,
    chunks: list[str],
    model,
    embedding_index,
    weight_sparse: float,
    k: int = 5,
    reranker_cutoff: int = 20  # Number of top results to rerank
):
    # Get BM25 results
    tokenized_query = nltk.word_tokenize(query)
    bm25_scores = np.array(bm25.get_scores(tokenized_query))  # Already numpy array
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:reranker_cutoff]

    # Get dense results using OpenAI embeddings
    query_embedding = model.embed_query(query)

    # Query Pinecone index
    dense_results = embedding_index.query(
        vector=query_embedding,
        top_k=reranker_cutoff,
        include_values=True
    )

    # Extract scores and indices from Pinecone results and convert to numpy arrays
    dense_scores = np.array([match['score'] for match in dense_results['matches']])
    dense_indices = np.array([int(match['id']) for match in dense_results['matches']])

    # Normalize scores (now all operations use numpy)
    bm25_scores_norm = (bm25_scores[bm25_top_indices] - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
    dense_scores_norm = (dense_scores - np.min(dense_scores)) / (np.max(dense_scores) - np.min(dense_scores))

    # Create combined results
    combined_results = {}

    # Add BM25 results
    for idx, score in zip(bm25_top_indices, bm25_scores_norm):
        combined_results[idx] = {'score': weight_sparse * score, 'count': 1}

    # Add dense results
    for idx, score in zip(dense_indices, dense_scores_norm):
        if idx in combined_results:
            combined_results[idx]['score'] += (1 - weight_sparse) * score
            combined_results[idx]['count'] += 1
        else:
            combined_results[idx] = {'score': (1 - weight_sparse) * score, 'count': 1}

    # Calculate final scores
    for idx in combined_results:
        combined_results[idx]['final_score'] = combined_results[idx]['score'] / combined_results[idx]['count']

    # Sort by final score
    sorted_results = sorted(combined_results.items(), key=lambda x: x[1]['final_score'], reverse=True)

    # Return top k results with their chunks
    final_results = []
    for idx, scores in sorted_results[:k]:
        final_results.append({
            'id': str(idx),
            'score': scores['final_score'],
            'metadata': {'text': chunks[idx]}
        })

    return final_results

"""# **Evaluate Rag**"""

from tqdm import tqdm
import pandas as pd
import bert_score # Import bert_score

def evaluate_rag_system(
    best_answers_df: pd.DataFrame,
    bm25,
    chunks: list[str],
    embedding_model,
    embedding_index,
    generate_amswer,
    weight_sparse: float,
    n_samples: int = None,  # Optional: limit number of samples for testing
    reranker_cutoff: int = 20
):


    # Initialize results storage
    results = []

    # Get subset of dataframe if n_samples is specified
    eval_df = best_answers_df.head(n_samples) if n_samples else best_answers_df

    # Lists to store all references and candidates for batch BERTScore computation
    all_references = []
    all_candidates = []

    # Iterate through questions and answers
    for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Evaluating Questions"):
        query = row['question']
        reference_answer = row['answer']

        try:
            # Get relevant context using fusion ranking
            retrieved_results = fusion_rank_search(
                query=query,
                bm25=bm25,
                chunks=chunks,
                model=embedding_model,
                embedding_index=embedding_index,
                k=5,
                weight_sparse=0.1,
                reranker_cutoff=reranker_cutoff
            )

            # Prepare pairs for reranking
            pairs = [(query, result['metadata']['text']) for result in retrieved_results]

            # Get reranker scores - use them directly for final ranking
            rerank_scores = get_reranker_score(pairs)

            # Update results with reranker scores
            for result, rerank_score in zip(retrieved_results, rerank_scores):
                result['metadata']['rerank_score'] = float(rerank_score)
                # Use reranker score as the final score
                result['score'] = float(rerank_score)

            # Resort based on reranker scores
            retrieved_results.sort(key=lambda x: x['score'], reverse=True)

            # Prepare context for LLM
            context = "\n".join([res['metadata']['text'] for res in retrieved_results])

            # Generate answer using LLM
            generated_answer = generate_amswer(context, query)

            # Store answers for batch BERTScore computation
            all_references.append(reference_answer)
            all_candidates.append(generated_answer)

            # Store intermediate results
            result = {
                'question': query,
                'reference_answer': reference_answer,
                'generated_answer': generated_answer,
                'retrieved_contexts': [res['metadata']['text'] for res in retrieved_results],
                'context_scores': [res['score'] for res in retrieved_results]
            }
            results.append(result)

        except Exception as e:
            print(f"Error processing question {idx}: {str(e)}")
            continue

    # Calculate BERTScore for all pairs at once
    P, R, F1 = bert_score.score(
        all_candidates,
        all_references,
        lang="en",
        verbose=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Add BERTScore metrics to results
    for idx, (p, r, f1) in enumerate(zip(P, R, F1)):
        results[idx].update({
            'bertscore_precision': p.item(),
            'bertscore_recall': r.item(),
            'bertscore_f1': f1.item()
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate and print average scores
    avg_scores = {
        'Average BERTScore Precision': results_df['bertscore_precision'].mean(),
        'Average BERTScore Recall': results_df['bertscore_recall'].mean(),
        'Average BERTScore F1': results_df['bertscore_f1'].mean()
    }

    return results_df, avg_scores

def print_evaluation_results(results_df, avg_scores):
    print("\nAverage Scores:")
    for metric, score in avg_scores.items():
        print(f"{metric}: {score:.4f}")

    print("\nDetailed Results Sample (first 3):")
    for idx, row in results_df.head(3).iterrows():
        print("\nQuestion:", row['question'])
        print("Reference Answer:", row['reference_answer'])
        print("Generated Answer:", row['generated_answer'])
        print(f"BERTScore Precision: {row['bertscore_precision']:.4f}")
        print(f"BERTScore Recall: {row['bertscore_recall']:.4f}")
        print(f"BERTScore F1: {row['bertscore_f1']:.4f}")
        # print("\nRetrieved Contexts:")
        # for context, score in zip(row['retrieved_contexts'], row['context_scores']):
        #     print(f"Score: {score:.4f}")
        #     print(f"Context: {context[:200]}...")

"""# **Compare Rag Evaluations**"""

from typing import Tuple

def compare_rag_evaluations(best_answers_df: pd.DataFrame,
                          set1_params: dict,
                          set2_params: dict,
                          generate_amswer,
                          weight_sparse: float,
                          n_samples: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compare RAG evaluation results between two parameter sets.

    Args:
        best_answers_df: DataFrame with questions and answers
        set1_params: Dictionary with parameters for first evaluation
        set2_params: Dictionary with parameters for second evaluation
        llm_chain: The LLM chain to use for evaluation
        n_samples: Optional number of samples to evaluate

    Returns:
        DataFrame with comparison results
    """
    # Run evaluations for both sets
    results1_df, avg_scores1 = evaluate_rag_system(
        best_answers_df=best_answers_df,
        weight_sparse=weight_sparse,
        bm25=set1_params['bm25'],
        chunks=set1_params['chunks'],
        embedding_model=set1_params['embedding_model'],
        embedding_index=set1_params['embedding_index'],
        generate_amswer=generate_amswer,
        n_samples=n_samples
    )

    print_evaluation_results(results1_df, avg_scores1)

    results2_df, avg_scores2 = evaluate_rag_system(
        best_answers_df=best_answers_df,
        weight_sparse=weight_sparse,
        bm25=set2_params['bm25'],
        chunks=set2_params['chunks'],
        embedding_model=set2_params['embedding_model'],
        embedding_index=set2_params['embedding_index'],
        generate_amswer=generate_amswer,
        n_samples=n_samples
    )

    print_evaluation_results(results2_df, avg_scores2)
    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'Metric': ['BERTScore Precision', 'BERTScore Recall', 'BERTScore F1'],
        'Contextual': [
            avg_scores1['Average BERTScore Precision'],
            avg_scores1['Average BERTScore Recall'],
            avg_scores1['Average BERTScore F1']
        ],
        'Regular': [
            avg_scores2['Average BERTScore Precision'],
            avg_scores2['Average BERTScore Recall'],
            avg_scores2['Average BERTScore F1']
        ]
    })

    # Calculate differences
    comparison['Difference'] = comparison['Contextual'] - comparison['Regular']

        # Calculate differences
    comparison['Difference'] = comparison['Contextual'] - comparison['Regular']

    # Calculate percentage difference
    # Formula: ((new - old) / old) * 100
    comparison['Difference %'] = ((comparison['Contextual'] - comparison['Regular']) / comparison['Regular'] * 100).round(2)

    # Format numbers to 4 decimal places
    for col in ['Contextual', 'Regular', 'Difference', 'Difference %']:
        comparison[col] = comparison[col].round(4)

    return comparison, results1_df, results2_df

"""# **Defining Answer Generation Chain**

## **OpenAi**
"""

prompt_template_answer = ChatPromptTemplate.from_messages([
    ("system",
            """You are an AI assistant specialized in answering user queries based solely on provided context. Your primary goal is to provide clear, concise, and relevant answers without adding, making up, or hallucinating any information.
            """
     ),
    ("human","""Now, consider the following context carefully:
      <context>
      {context}
      </context>

      Here is the user's query:
      <query>
      {query}
      </query>

      Before answering, please follow these steps:

      1. Analyze the user's query and the provided context:
        a. Identify the key elements of the user's query.
        b. Find and quote relevant information from the context.
        c. Explicitly link the quoted information to the query elements.
        d. Formulate a potential answer based only on the context.
        e. Explicitly check that your answer doesn't include any information not present in the context.
        f. If the context doesn't contain enough information to answer the query, note this.

      2. After your analysis process, provide your final answer or response. Do not include your analysis steps in your final answer or response, only the result.

      If the context does not contain enough information to answer the user's query confidently and accurately, your final response should be: "I do not have enough information to answer this question based on the provided context."

      Remember, it's crucial that your answer is based entirely on the given context. Do not add any external information or make assumptions beyond what is explicitly stated in the context.

    """)
])

from langchain_core.output_parsers import StrOutputParser

def create_answer_chain(llm):
  return prompt_template_answer | llm | StrOutputParser()

def get_generate_amswer(llm_chain):
    def generate_amswer(context, query):
        llm_response = llm_chain.invoke({
                    "context": context,
                    "query": query
                })
        return llm_response.content if hasattr(llm_response, 'content') else llm_response
    return generate_amswer

"""# **Creating Answer Generation chain**"""

answer_chain = create_answer_chain(llm)

embedding_index_contextual= pc.Index(EMBEDDING_INDEX_CONTEXTUAL)
embedding_index_regular= pc.Index(EMBEDDING_INDEX_REGULAR)

"""# **Running the RAG**"""

set1_params = {
    'embedding_index': embedding_index_contextual,
    'chunks': chunks_with_context,
    'bm25': bm25_contextual,
    'embedding_model': embedding_model  # Add your embedding model here
}

set2_params = {
    'embedding_index': embedding_index_regular,
    'chunks': chunks_regular,
    'bm25': bm25_regular,
    'embedding_model': embedding_model  # Add your embedding model here
}

# Run comparison
comparison_results,results1_df, results2_df = compare_rag_evaluations(
    best_answers_df=best_answers_df,
    weight_sparse=0.3, #alpha
    set1_params=set1_params,
    set2_params=set2_params,
    generate_amswer=get_generate_amswer(answer_chain),
    n_samples=None  # Set to a number if you want to limit samples
)

# Display results as markdown table
print(comparison_results.to_markdown(index=False))
# Save DataFrame to CSV

"""# **Saving Results to files**"""

comparison_results_file = create_filename_timestamp(filename='comparison_results', extension="csv")
comparison_results.to_csv(comparison_results_file, index=False)
results1_df_file = create_filename_timestamp(filename='contextual_rag_results', extension="csv")
results1_df.to_csv(results1_df_file, index=False)
results2_df_file = create_filename_timestamp(filename='regular_rag_results', extension="csv")
results2_df.to_csv(results2_df_file, index=False)

files.download(comparison_results_file)
files.download(results1_df_file)
files.download(results2_df_file)