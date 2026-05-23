1. AI Agent - LangChain - Insurance
   Covers Document Loading, Text Splitting, Sentence Transformer, Embeddings & Vector Store via ChromaDB.
   Chunking Strategy Design Decisions, Embedding Model Selection, 
   LLM - Basic retrieval, Finalize Retreival Strategy, Conversation Memory for multi turn conversations
   LangChain - Natural Conversation setup, Multi turn conversation, Role Definitions, Constraints, Fallback
   
3. RAG-Agent-Helpmate
  Covers end-to-end design and implementation of a comprehensive conversational AI system for insurance policy documents 
  using a pure Lang Chain architecture with FAISS vector store and Groq’s llama-3.3-70b LLM, emphasizing robust document processing, 
  semantic search, multi-turn conversation memory, accurate source citation, and an interactive user interface, while highlighting 
  technical decisions, challenges addressed, performance benchmarks, business benefits, and future enhancements.

4. Shop Assist:
  ShopAssist AI is a chatbot designed to assist users in online laptop shopping by parsing a laptop dataset to provide personalized
recommendations based on six key user requirements, utilizing large language models and function calling APIs to improve accuracy,
 conversational dynamics, and performance while addressing challenges such as model dependency and cost, ultimately achieving significant
improvements in API efficiency, response time, success rate, and user experience.

Flow:

1.	Data Load & parsing from source: Load data and parse based on RegEx for feature extraction 
2.	Function Calling API for Laptop Search 
3.	API Communication via chat completion with exponential backoZ 
4.	Moderation Check 
5.	Laptop Search Function with feature extraction 
6.	Conversation  
7.	Manual Mode – in case of rate limit  
8.	Main Dialogue Function 
9.	Helper Function to fetch requirements from conversation 
10.	Main Execution

