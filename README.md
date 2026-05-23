# AI Agent Projects

1. Insurance Policy Assistant - RAG with FAISS

Built a conversational AI agent for insurance policy Q&A using pure LangChain architecture with FAISS vector store.

Problem: Insurance policies are complex and difficult to understand, requiring automated Q&A with accurate source citations

Implementation:
- Pure LangChain architecture (no ChromaDB dependency)
- FAISS vector store with 166 document chunks
- Hybrid chunking: section-based + fixed-size with 200-word overlap
- HuggingFace embeddings: multi-qa-mpnet-base-dot-v1 (768-dim)
- Groq LLM: llama-3.3-70b-versatile
- History-aware retriever for context reformulation
- Conversational memory with chat history tracking
- Custom PDF processing with pdfplumber for table extraction

Results: Accurate retrieval with automatic source citations (policy name + page). Context-aware multi-turn conversations.

Technical Details:
- RAG Pipeline: PDF extraction -> Chunking -> Embedding -> FAISS indexing -> Retrieval -> LLM generation
- Vector Search: FAISS with cosine similarity, top-5 retrieval
- Chunking: Hybrid section-based + fixed-size with overlap
- LLM Config: temperature=0.3, max_tokens=2048

Stack: Python, LangChain, FAISS, Groq API, HuggingFace, pdfplumber, sentence-transformers

2. Insurance Policy Assistant - RAG with ChromaDB

Enhanced version with ChromaDB vector database and advanced retrieval techniques.

Problem: Multi-document insurance policy search with improved retrieval quality

Implementation:
- ChromaDB persistent storage with custom embedding function
- Hybrid chunking strategy: 400-word chunks, 50-word overlap
- Q&A optimized embeddings: multi-qa-mpnet-base-dot-v1
- 80 chunks from 78 pages across multiple policy documents
- Metadata tracking: policy name, page number, chunk ID, word count
- Cross-encoder reranking for improved relevance
- Regex-based PDF parsing for structured data extraction

Results: Improved retrieval quality with semantic chunking. Distance-based relevance scoring for top-K results.

Technical Details:
- RAG Pipeline: PDF extraction -> Regex parsing -> Chunking -> Embedding -> ChromaDB storage -> Retrieval -> LLM generation
- Vector Search: ChromaDB with cosine similarity, top-5 retrieval with distance scoring
- Chunking: Hybrid 400-word chunks with 50-word overlap
- Cross-encoder reranking: ms-marco-MiniLM-L-6-v2

Stack: Python, LangChain, ChromaDB, Groq API, sentence-transformers, pdfplumber, pandas

3. ShopAssist - Laptop Recommender

Conversational AI using Groq Function Calling API for intelligent laptop recommendations.

Problem: Users struggle to match technical specifications with their actual needs and budget

Implementation:
- Groq Function Calling API with structured parameter extraction
- 6 requirement parameters: GPU, display, portability, multitasking, processing, budget
- Regex-based feature extraction from laptop descriptions
- Multi-criteria scoring system with weighted matching
- Tenacity retry logic with exponential backoff (5 attempts, 10-120s wait)
- Manual fallback mode for rate limit scenarios
- Groq LLM: llama-3.3-70b-versatile for conversation, llama-3.1-8b-instant for moderation
- Dataset: 20 laptops with price range Rs.25,000 - Rs.280,000

Results: Natural conversation flow with automatic requirement extraction. Top-3 recommendations with match scores.

Technical Details:
- Conversation Pipeline: User input -> Moderation -> Function calling -> Parameter extraction -> Laptop search -> Scoring -> LLM response
- Function Calling: Structured extraction with 6 enum-constrained parameters
- Scoring System: Multi-criteria matching (5 features) with weighted scoring
- Error Handling: Tenacity retry (5 attempts, exponential backoff 10-120s), manual fallback mode
- LLM Config: temperature=0.7 for conversation, temperature=0 for moderation

Stack: Python, Groq API, Pandas, Tenacity, regex
