# Learning LangChain: A Journey

Welcome to the repository documenting a progressive journey in learning and mastering LangChain! This project serves as a structured collection of examples, tutorials, and scripts that explore the various components of the LangChain framework.

## Table of Contents
1. [Prerequisites & Setup](#prerequisites--setup)
2. [Project Structure](#project-structure)
   - [1. Models](#1-models)
   - [2. Prompts](#2-prompts)
   - [3 & 4. Structured Output](#3--4-structured-output-parsers)
   - [5. Chains](#5-chains)
   - [6 & 7. Runnables & LCEL](#6--7-runnables--lcel)
   - [8. Document Loader](#8-document-loader)
   - [9. Text Splitter](#9-text-splitter)
   - [10. Vector Store](#10-vector-store)
   - [11. Retrievers](#11-retrievers)
3. [Technologies & Integrations](#technologies--integrations)
4. [Learning Path](#learning-path)

---

## Prerequisites & Setup

To run the examples in this repository, you need Python installed on your system.

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd LangChain
   ```

2. **Set up a virtual environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory to store your API keys as many scripts rely on them to connect to LLM providers:
   ```env
   HUGGINGFACEHUB_API_TOKEN=your_token_here
   GOOGLE_API_KEY=your_token_here
   OPENAI_API_KEY=your_token_here
   ANTHROPIC_API_KEY=your_token_here
   ```

---

## Project Structure

The project is divided into sequentially numbered folders representing the foundational concepts of LangChain.

### 1. Models (`01_langchain_model`)
This section covers how to initialize and interact with different types of Language Models.

**Submodules:**
- **ChatModels:** Interfacing with Chat-based LLMs
  - `gemini_api.py` - Google Gemini integration with conversation memory
  - `HF_API.py` - HuggingFace API for text generation
  - `HF_Local.py` - Local HuggingFace model inference
- **EmbeddingModels:** Generating text embeddings
  - `hf_embedding.py` - Create embeddings using HuggingFace
  - `document_similarity.py` - Compare documents using embedding similarity

**Key Concepts:**
- Initializing different model types (ChatModel, EmbeddingModel)
- Managing conversation history with System/Human/AI messages
- Computing semantic similarity between documents

---

### 2. Prompts (`02_prompt`)
Exploring how to craft and manage prompts for LLMs.

**Files:**
- `prompt_generator.py` - Dynamic prompt generation
- `prompt_ui.py` - Interactive prompt interface
- `chat_prompt_template.py` - Template-based chat prompts
- `chatbot.py` - Simple chatbot implementation
- `message_placeholder.py` - Managing message placeholders

**Key Concepts:**
- Prompt templates with variables
- Chat prompt templates with role-based messages (System, Human, AI)
- Building conversational context
- Dynamic prompt construction

---

### 3 & 4. Structured Output & Parsers (`03_Structured_Output`, `04_Structure_output_2`)
Focuses on ensuring predictability in LLM outputs.

**Module 3:**
- `typeddict.py` - Using Python TypedDict for structured schemas

**Module 4:**
- `StrOutputParser.py` - Basic string output parsing
- `JsonOutputParser.py` - Parse LLM output to JSON
- `PydanticOuputParser.py` - Pydantic models for validation
- `StructuredOuputParser.py` - Custom structured output schemas
- `test.py` - Test implementations

**Key Concepts:**
- Output parsers for predictable formats
- JSON schema enforcement
- Pydantic validation for type safety
- Custom output schemas with TypedDict

---

### 5. Chains (`05_Chains`)
Using LangChain's chaining capabilities to build multi-step workflows.

**Files:**
- `simple_chain.py` - Basic single sequential chain
- `sequential_chain.py` - Passing outputs between chains
- `parallel_chain.py` - Executing multiple chains simultaneously
- `conditional_chain.py` - Branching logic based on conditions

**Key Concepts:**
- Chain composition
- Sequential data flow
- Parallel execution patterns
- Conditional branching in workflows

---

### 6 & 7. Runnables & LCEL (`06_Runnables_basic`, `07_Runnables_Primitives`)
Introduction to the **LangChain Expression Language (LCEL)** - the modern, modular approach to chaining.

**Module 6:**
- `dummy_runnable.ipynb` - Interactive Jupyter notebook exploration

**Module 7:**
- `RunnableLambda.py` - Custom functions as runnables
- `Passthrough_runnable.py` - Pass data through unchanged
- `Sequential_runnable.py` - Chain components sequentially
- `Parallel_runnable.py` - Execute multiple branches in parallel
- `Branch_Runnable.py` - Conditional branching

**Key Concepts:**
- LCEL syntax with pipe operator (`|`)
- RunnableSequence for explicit chaining
- RunnableLambda for custom transformations
- Parallel and sequential composition
- RunnablePassthrough for data flow control

---

### 8. Document Loader (`08_Document_Loader`)
Loading documents from various sources for RAG applications.

**Files:**
- `text_loader.py` - Load plain text files
- `pypdf_loader.py` - Load PDF documents
- `csv_loader.py` - Load CSV data
- `directory_loader.py` - Load all files from a directory
- `webbase_loader.py` - Scrape content from websites

**Key Concepts:**
- Document loaders as data ingestion layer
- Handling different file formats (txt, pdf, csv)
- Web scraping for content
- Directory batch loading
- Document metadata preservation

---

### 9. Text Splitter (`09_text_splitter`)
Breaking large documents into manageable chunks for processing.

**Files:**
- `length_based.py` - Split by character/token count
- `text_structure_based.py` - Split by paragraphs/sentences
- `document_structure_based.py` - Split by document hierarchy
- `semantic_meaning_based.py` - Split by semantic similarity

**Key Concepts:**
- Chunk size and overlap configuration
- Recursive character splitting
- Preserving context across chunks
- Semantic-aware splitting for coherence

---

### 10. Vector Store (`10_vector_store`)
Storing and indexing document embeddings for semantic search.

**Files:**
- `vector_store.ipynb` - Interactive vector store implementation
- `chroma_db/` - Persistent ChromaDB storage

**Key Concepts:**
- Embedding documents into vector space
- ChromaDB as lightweight vector store
- Similarity search fundamentals
- Persistent storage of embeddings
- Querying by semantic similarity

---

### 11. Retrievers (`11_Retrievers`)
Retrieval mechanisms for fetching relevant context in RAG applications.

**Based on DataSource:**
- `wikipedia_retriever.ipynb` - Wikipedia API integration
- `vectorstore_retriever.ipynb` - Vector store backed retrieval

**Based on Retrieval Mechanism:**
- `MaximumMarginalRelevance.ipynb` - MMR for diversity
- `MultiQueryRetriever.ipynb` - Multiple query reformulation
- `ContextualCompressionRetriever.ipynb` - Compress retrieved results

**Key Concepts:**
- Vector store retrieval
- Wikipedia as external knowledge source
- Maximum Marginal Relevance (MMR) for diverse results
- Multi-query approach for robustness
- Contextual compression for concise results

---

## Technologies & Integrations

The repository is built leveraging several modern AI tools and libraries:

**Core Framework:**
- LangChain, langchain-core, langchain-community, langchain-experimental

**LLM Integrations:**
- OpenAI (`langchain-openai`)
- Anthropic (`langchain-anthropic`)
- Google Gemini (`langchain-google-genai`)
- HuggingFace (`langchain-huggingface`, `transformers`)

**Vector Stores & Retrieval:**
- ChromaDB
- Wikipedia API

**Document Processing:**
- PyPDF (PDF parsing)
- BeautifulSoup4 (HTML parsing)

**Utilities:**
- python-dotenv (environment management)
- NumPy (numerical operations)
- Scikit-learn (ML utilities)

---

## Learning Path

This repository follows a **progressive learning journey**:

```
Models → Prompts → Structured Output → Chains → LCEL → Documents → Splitters → Vectors → Retrievers
```

1. **Start with Models** - Learn to initialize and interact with LLMs
2. **Master Prompts** - Craft effective prompts with templates
3. **Structure Outputs** - Ensure predictable, parseable responses
4. **Build Chains** - Connect multiple operations sequentially
5. **Embrace LCEL** - Modern, composable pipeline syntax
6. **Load Documents** - Ingest data from various sources
7. **Split Text** - Chunk documents for processing
8. **Create Vectors** - Embed and store semantic representations
9. **Retrieve Context** - Build RAG applications with retrievers

---

## Running Examples

Most examples can be run directly:

```bash
# Run a simple chatbot
python 02_prompt/chatbot.py

# Test structured output
python 04_Structure_output_2/JsonOutputParser.py

# Load a document
python 08_Document_Loader/text_loader.py
```

Ensure your `.env` file is properly configured with API keys before running.

---

Happy building!
