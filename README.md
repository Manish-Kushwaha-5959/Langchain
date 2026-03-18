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
3. [Technologies & Integrations](#technologies--integrations)

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

### 1. Models (`1_langchain_model`)
This section covers how to initialize and interact with different types of Language Models.
- **ChatModels:** Interfacing with Chat-based LLMs like Google Gemini (`gemini_api.py`) and HuggingFace (`HF_API.py`, `HF_Local.py`).
- **EmbeddingModels:** Generating text embeddings and working with document similarity (`hf_embedding.py`, `document_similarity.py`).

### 2. Prompts (`2_prompt`)
Exploring how to craft and manage prompts for LLMs.
- Demonstrates **Prompt Templates**, **Chatbot contexts**, and dynamic prompt generation.
- Features scripts like `prompt_generator.py`, `chat_prompt_template.py`, and a `chatbot.py`.

### 3 & 4. Structured Output & Parsers (`3_Structured_Output`, `4_Structure_output_2`)
Focuses on ensuring predictability in LLM outputs.
- Shows how to format LLM responses into strict structures like JSON or custom objects.
- Covers multiple parsers including `JsonOutputParser`, `PydanticOutputParser`, and `StrOutputParser`.
- Demonstrates formatting output using `TypedDict` in Python.

### 5. Chains (`5_Chains`)
Using LangChain's chaining capabilities to build multi-step workflows.
- Contains examples for building traditional chains:
  - `simple_chain.py`: A basic single sequential logic.
  - `sequential_chain.py`: Passing outputs from one chain to another.
  - `parallel_chain.py`: Executing multiple chains simultaneously.
  - `conditional_chain.py`: Branching logic based on conditions.

### 6 & 7. Runnables & LCEL (`6_Runnables_basic`, `7_Runnables_Primitives`)
An introduction to the **LangChain Expression Language (LCEL)**, showcasing modern and modular approaches to chaining.
- Introduces fundamental primitives used to build complex customized workflows seamlessly.
- Examples include `RunnableLambda.py`, `Passthrough_runnable.py`, `Sequential_runnable.py`, and `Parallel_runnable.py`. 
- Includes a Jupyter notebook (`dummy_runnable.ipynb`) for interactive exploration.

---

## Technologies & Integrations

The repository is built leveraging several modern AI tools and libraries, explicitly:
- **LangChain Core Framework**
- **LLM Integrations:** OpenAI, Anthropic, Google Gemini (PaLM), HuggingFace
- **Utilities:** Python-dotenv (env management), NumPy, Scikit-learn

Happy building! 🚀