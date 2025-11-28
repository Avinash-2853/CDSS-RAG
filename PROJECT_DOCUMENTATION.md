# CDSS-RAG: Clinical Decision Support System with Retrieval-Augmented Generation

## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system for clinical decision support, specifically designed to generate medical summaries from doctor-patient conversations. The system uses Large Language Models (LLMs) enhanced with relevant clinical context retrieved from a medical knowledge base.

### Key Components

1. **Data Preprocessing Pipeline**: Processes and indexes medical documents into a vector database
2. **RAG-Based Summarization**: Generates clinical summaries using LLAMA and Gemini models
3. **Comprehensive Evaluation**: Assesses both generation quality and retrieval performance

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Prerequisites and Setup](#prerequisites-and-setup)
3. [Step-by-Step Workflow](#step-by-step-workflow)
4. [Detailed Component Descriptions](#detailed-component-descriptions)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Output Files](#output-files)
7. [Troubleshooting](#troubleshooting)

---

## Project Structure

```
CDSS-RAG/
├── data/
│   ├── raw/                          # Raw input data
│   │   ├── conversation_summary.csv  # Doctor-patient conversations
│   │   ├── medical_data.csv          # Medical text data
│   │   ├── patient_notes.csv         # Patient notes
│   │   ├── PMC-Patients.csv          # PMC patient data
│   │   └── pubmed_dataset.csv        # PubMed articles
│   └── processed/                    # Processed outputs
│       ├── conversation_summary_using_llama.csv
│       ├── conversation_summary_using_gemini.csv
│       ├── evaluated_llama.csv
│       ├── evaluated_gemini.csv
│       └── model_comparison_summary.csv
├── notebooks/
│   ├── preprocess.ipynb              # Data preprocessing and vector DB creation
│   ├── summarization_using_llama.ipynb  # LLAMA-based summarization
│   ├── summarization_using_gemini.ipynb # Gemini-based summarization
│   ├── evaluation.ipynb              # Model evaluation
│   └── chroma_store/                 # ChromaDB vector database
└── primock57/                        # Additional dataset resources
```

---

## Prerequisites and Setup

### Required Libraries

```python
# Core libraries
pandas
numpy
langchain
langchain-community
langchain-groq  # For LLAMA via Groq
google-generativeai  # For Gemini

# NLP and embeddings
transformers
torch
sentence-transformers
scikit-learn

# Evaluation metrics
nltk
rouge-score
bert-score

# Vector database
chromadb

# Medical NLP
spacy
scispacy
```

### API Keys Required

1. **Groq API Key** (for LLAMA model)
   - Set in `.env` file: `GROQ_API_KEY=your_key_here`

2. **Google API Key** (for Gemini model)
   - Set in `.env` file: `GOOGLE_API_KEY=your_key_here`

### Installation Steps

```bash
# Install Python dependencies
pip install pandas numpy langchain langchain-community langchain-groq
pip install google-generativeai transformers torch sentence-transformers
pip install scikit-learn nltk rouge-score bert-score chromadb
pip install spacy scispacy

# Download spaCy model
python -m spacy download en_core_sci_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

---

## Step-by-Step Workflow

### Phase 1: Data Preprocessing (`preprocess.ipynb`)

**Objective**: Create a searchable medical knowledge base from multiple data sources.

#### Step 1.1: Load Raw Data
- Loads first 1,000 rows from each medical dataset:
  - `medical_data.csv` (column: `TEXT`)
  - `patient_notes.csv` (column: `pn_history`)
  - `PMC-Patients.csv` (column: `patient`)
  - `pubmed_dataset.csv` (column: `contents`)

#### Step 1.2: Text Cleaning and Normalization
Performs the following cleaning operations:
- **Whitespace normalization**: Removes extra spaces
- **Special character removal**: Removes non-alphanumeric characters (except basic punctuation)
- **Abbreviation expansion**: Expands medical abbreviations using UMLS-style mapping
  - Examples: `HTN` → `hypertension`, `DM` → `diabetes mellitus`, `SOB` → `shortness of breath`
- **Lowercasing**: Converts all text to lowercase
- **PHI De-identification**: Redacts Protected Health Information:
  - Email addresses
  - Phone numbers
  - Dates
  - Medical Record Numbers (MRN)
  - Names

#### Step 1.3: Document Merging
- Combines all cleaned text from different sources into a master document
- Preserves source metadata for traceability

#### Step 1.4: Text Chunking
- Uses `RecursiveCharacterTextSplitter` from LangChain
- Splits documents into chunks of ~100-300 words
- Maintains semantic coherence within chunks

#### Step 1.5: Embedding Generation
- Uses **Bio_ClinicalBERT** (`emilyalsentzer/Bio_ClinicalBERT`) for embeddings
- Generates 768-dimensional embeddings for each chunk
- Optimized for biomedical/clinical text understanding

#### Step 1.6: ChromaDB Storage
- Stores embeddings and chunks in ChromaDB vector database
- Persists to: `notebooks/chroma_store/`
- Enables fast semantic similarity search

**Output**: Vector database ready for RAG retrieval

---

### Phase 2: RAG-Based Summarization

#### Option A: LLAMA Summarization (`summarization_using_llama.ipynb`)

**Objective**: Generate medical summaries using LLAMA-3.3-70B model with RAG.

##### Step 2.1: Load Conversation Data
```python
df = pd.read_csv("data/raw/conversation_summary.csv")
df = df.head(15)  # Process first 15 conversations
```

##### Step 2.2: Initialize LLM
- Model: `llama-3.3-70b-versatile` via Groq API
- Temperature: 0.2 (for consistent outputs)
- Max tokens: 2048

##### Step 2.3: Load Vector Database
- Loads ChromaDB from `notebooks/chroma_store/`
- Uses same Bio_ClinicalBERT embeddings for consistency
- Creates retriever with default similarity search

##### Step 2.4: Create RAG Chain
```python
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)
```

**RAG Process Flow**:
1. **Retrieval**: For each conversation, retrieves top-k relevant chunks from ChromaDB
2. **Augmentation**: Combines retrieved context with conversation text
3. **Generation**: LLM generates summary using both conversation and context

##### Step 2.5: Generate Summaries
- Iterates through each conversation
- Invokes RAG chain for each conversation
- Stores generated summaries in `rag_summary` column

##### Step 2.6: Save Results
- Saves to: `data/processed/conversation_summary_using_llama.csv`
- Columns: `conversation`, `summary` (gold standard), `rag_summary` (generated)

#### Option B: Gemini Summarization (`summarization_using_gemini.ipynb`)

**Objective**: Generate medical summaries using Gemini-2.5-Flash model with RAG.

**Process**: Similar to LLAMA, but uses:
- Model: `gemini-2.5-flash`
- Custom `GeminiRunnable` wrapper for LangChain compatibility
- Output: `data/processed/conversation_summary_using_gemini.csv`

---

### Phase 3: Evaluation (`evaluation.ipynb`)

**Objective**: Comprehensively evaluate both generation quality and retrieval performance.

#### Step 3.1: Load Evaluation Data
- Loads both LLAMA and Gemini summary outputs
- Extracts retrieved chunks from ChromaDB for each conversation

#### Step 3.2: Generation Metrics

##### BLEU Score
- **Purpose**: Measures n-gram overlap between generated and reference summaries
- **Method**: Sentence-level BLEU with smoothing
- **Range**: 0.0 to 1.0 (higher is better)

##### ROUGE-L Score
- **Purpose**: Measures longest common subsequence (LCS) overlap
- **Method**: F1 score of LCS-based matching
- **Range**: 0.0 to 1.0 (higher is better)

##### BERTScore
- **Purpose**: Semantic similarity using contextual embeddings
- **Model**: Bio_ClinicalBERT or PubMedBERT
- **Method**: Cosine similarity of token embeddings
- **Range**: 0.0 to 1.0 (higher is better)

#### Step 3.3: Retrieval Metrics

##### Precision@5
- **Formula**: `(Relevant chunks in top 5) / 5`
- **Purpose**: Measures relevance of top retrieved chunks
- **Relevance Threshold**: Cosine similarity ≥ 0.70 (using ClinicalBERT)

##### Recall@10
- **Formula**: `(Relevant chunks in top 10) / (Total relevant chunks)`
- **Purpose**: Measures coverage of relevant information
- **Note**: Total relevant chunks calculated from all 15 retrieved chunks

##### MRR (Mean Reciprocal Rank)
- **Formula**: `1 / (rank of first relevant chunk)`
- **Purpose**: Measures how quickly first relevant result appears
- **Range**: 0.0 to 1.0 (1.0 = first chunk is relevant)

#### Step 3.4: Metric Computation
- Computes metrics for each conversation
- Aggregates to mean values for model comparison
- Generates comparison summary

#### Step 3.5: Save Results
- `evaluated_llama.csv`: LLAMA results with all metrics
- `evaluated_gemini.csv`: Gemini results with all metrics
- `model_comparison_summary.csv`: Side-by-side comparison

---

## Detailed Component Descriptions

### 1. Embedding Model: Bio_ClinicalBERT

**Model**: `emilyalsentzer/Bio_ClinicalBERT`
- **Type**: BERT-based transformer
- **Dimensions**: 768
- **Training**: Trained on clinical notes from MIMIC-III
- **Use Case**: Optimal for medical/clinical text similarity

**Why This Model?**
- Domain-specific: Trained on clinical text
- Better semantic understanding of medical terminology
- Higher accuracy for clinical similarity tasks

### 2. Vector Database: ChromaDB

**Why ChromaDB?**
- Lightweight and easy to use
- Persistent storage
- Efficient similarity search
- LangChain integration

**Storage Structure**:
- Embeddings: 768-dimensional vectors
- Metadata: Source, chunk index, timestamps
- Text: Original chunk content

### 3. Retrieval Process

**Similarity Search**:
1. Embed query (conversation) using Bio_ClinicalBERT
2. Compute cosine similarity with all stored embeddings
3. Retrieve top-k most similar chunks (k=15)
4. Return chunks as context for LLM

**Relevance Determination**:
- Threshold: 0.70 cosine similarity
- Chunks with similarity ≥ 0.70 are considered "relevant"
- Used for precision/recall calculations

### 4. Prompt Engineering

**Template Structure**:
```
You are a medical summarization expert. Given:
- Conversation between doctor and patient
- Relevant clinical context (from RAG)

Generate a concise, accurate medical note including:
- History and symptoms
- Relevant negatives
- Medical/social/family history
- Impression
- Plan
```

**Key Features**:
- Explicit instructions for medical note format
- Example format provided
- Emphasis on accuracy and key clinical details
- Integration of RAG context

---

## Evaluation Metrics

### Generation Metrics Explained

#### BLEU Score
- **What it measures**: Token-level n-gram overlap
- **Strengths**: Fast, widely used
- **Limitations**: Doesn't capture semantic similarity
- **Interpretation**: 
  - 0.0-0.3: Poor overlap
  - 0.3-0.6: Moderate overlap
  - 0.6-1.0: Good overlap

#### ROUGE-L Score
- **What it measures**: Longest common subsequence (sentence-level)
- **Strengths**: Better for longer sequences
- **Limitations**: Still token-based
- **Interpretation**: Similar to BLEU

#### BERTScore
- **What it measures**: Semantic similarity using embeddings
- **Strengths**: Captures meaning, not just tokens
- **Limitations**: Computationally expensive
- **Interpretation**: 
  - 0.0-0.5: Low semantic similarity
  - 0.5-0.7: Moderate similarity
  - 0.7-1.0: High similarity

### Retrieval Metrics Explained

#### Precision@5
- **What it measures**: Fraction of top 5 chunks that are relevant
- **Example**: If 4 out of top 5 chunks are relevant → Precision@5 = 0.8
- **Interpretation**: Higher = more relevant results at top

#### Recall@10
- **What it measures**: Fraction of all relevant chunks found in top 10
- **Example**: If 5 relevant chunks exist total, and 3 are in top 10 → Recall@10 = 0.6
- **Interpretation**: Higher = better coverage of relevant information

#### MRR (Mean Reciprocal Rank)
- **What it measures**: How quickly first relevant result appears
- **Example**: 
  - First relevant at position 1 → MRR = 1.0
  - First relevant at position 3 → MRR = 0.333
  - No relevant results → MRR = 0.0
- **Interpretation**: Higher = faster to find relevant information

---

## Output Files

### 1. Summarization Outputs

**`conversation_summary_using_llama.csv`**
- Columns: `conversation`, `summary`, `rag_summary`
- Contains: Original conversations, gold summaries, LLAMA-generated summaries

**`conversation_summary_using_gemini.csv`**
- Columns: `conversation`, `summary`, `rag_summary`
- Contains: Original conversations, gold summaries, Gemini-generated summaries

### 2. Evaluation Outputs

**`evaluated_llama.csv`**
- All original columns plus:
  - `bleu_score`: BLEU score per conversation
  - `rouge_score`: ROUGE-L score per conversation
  - `bertscore`: BERTScore per conversation
  - `precision_5`: Precision@5 per conversation
  - `recall_10`: Recall@10 per conversation
  - `mrr`: MRR per conversation
  - `retrieved_chunks`: JSON string of retrieved chunks

**`evaluated_gemini.csv`**
- Same structure as `evaluated_llama.csv`

**`model_comparison_summary.csv`**
- Aggregated metrics comparing LLAMA vs Gemini:
  - Mean BLEU, ROUGE-L, BERTScore
  - Mean Precision@5, Recall@10, MRR
  - Side-by-side comparison

---

## Key Design Decisions

### 1. Why Bio_ClinicalBERT?
- **Domain-specific**: Trained on clinical text (MIMIC-III)
- **Better semantic understanding**: Understands medical terminology
- **Consistent embeddings**: Same model for indexing and evaluation

### 2. Why ChromaDB?
- **Simplicity**: Easy to set up and use
- **Persistence**: Data survives notebook restarts
- **Performance**: Efficient similarity search
- **Integration**: Native LangChain support

### 3. Why RAG?
- **Accuracy**: Provides factual context to LLM
- **Reduces hallucinations**: Grounds generation in real medical knowledge
- **Updatable**: Can update knowledge base without retraining LLM

### 4. Why Multiple Evaluation Metrics?
- **BLEU/ROUGE**: Standard NLP metrics (token-based)
- **BERTScore**: Semantic similarity (meaning-based)
- **Precision/Recall**: Retrieval quality assessment
- **MRR**: Ranking quality assessment

---

## Troubleshooting

### Common Issues

#### 1. ChromaDB Not Found
**Error**: `ChromaDB not found at specified path`
**Solution**: 
- Run `preprocess.ipynb` first to create the database
- Check path: `notebooks/chroma_store/`

#### 2. API Key Errors
**Error**: `API key not found`
**Solution**:
- Create `.env` file in project root
- Add: `GROQ_API_KEY=your_key` and `GOOGLE_API_KEY=your_key`
- Use `load_dotenv()` in notebooks

#### 3. Embedding Model Download Issues
**Error**: `Model not found`
**Solution**:
- Ensure internet connection for first-time download
- Models are cached after first download
- Check HuggingFace access

#### 4. Memory Issues
**Error**: `Out of memory`
**Solution**:
- Reduce `MAX_ROWS` in preprocessing (default: 1000)
- Process conversations in smaller batches
- Use CPU instead of GPU if needed

#### 5. Low Precision/Recall Scores
**Possible Causes**:
- Threshold too high (try 0.65 instead of 0.70)
- Embedding model mismatch
- Insufficient relevant chunks in database

**Solution**:
- Adjust similarity threshold
- Ensure same embedding model used throughout
- Add more relevant medical documents to database

---

## Performance Considerations

### Computational Requirements

**Preprocessing**:
- Time: ~10-30 minutes (depending on data size)
- Memory: ~4-8 GB RAM
- Storage: ~500 MB for ChromaDB

**Summarization**:
- Time: ~1-2 minutes per conversation (with API calls)
- Cost: Depends on API pricing (Groq/Gemini)
- Rate limits: Check API provider limits

**Evaluation**:
- Time: ~5-10 minutes for 15 conversations
- Memory: ~2-4 GB RAM (for BERTScore)
- Most time-consuming: BERTScore computation

### Optimization Tips

1. **Batch Processing**: Process multiple conversations in parallel
2. **Caching**: Cache embeddings to avoid recomputation
3. **Selective Evaluation**: Evaluate subset if full evaluation is slow
4. **API Optimization**: Use batch API calls when available

---

## Future Enhancements

### Potential Improvements

1. **Better Chunking Strategy**:
   - Semantic chunking instead of fixed-size
   - Overlap between chunks for better context

2. **Hybrid Retrieval**:
   - Combine semantic search with keyword search
   - Re-ranking with cross-encoder models

3. **Fine-tuning**:
   - Fine-tune LLM on medical summaries
   - Domain-specific prompt optimization

4. **Multi-modal Support**:
   - Include images, lab results
   - Structured data integration

5. **Real-time Updates**:
   - Incremental database updates
   - Version control for knowledge base

---

## References and Resources

### Models Used
- **Bio_ClinicalBERT**: [HuggingFace](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)
- **LLAMA-3.3-70B**: [Groq](https://groq.com/)
- **Gemini-2.5-Flash**: [Google AI](https://ai.google.dev/)

### Libraries
- **LangChain**: [Documentation](https://python.langchain.com/)
- **ChromaDB**: [Documentation](https://www.trychroma.com/)
- **BERTScore**: [Paper](https://arxiv.org/abs/1904.09675)

### Medical Datasets
- **MIMIC-III**: Clinical notes database
- **PubMed**: Medical literature
- **PMC**: Patient case studies

---

## Contact and Support

For questions or issues:
1. Check this documentation first
2. Review notebook comments and code
3. Check error messages for specific guidance

---

## Version History

- **v1.0**: Initial implementation
  - Basic RAG pipeline
  - LLAMA and Gemini support
  - Comprehensive evaluation metrics

---

**Last Updated**: 2025-01-27

**Documentation Version**: 1.0

