# TaxGPT

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-FF4B4B.svg?logo=streamlit&style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-0.0.0-2ea44f.svg?logo=markdown&style=for-the-badge)
![PDFPlumberLoader](https://img.shields.io/badge/PDFPlumberLoader-Community-blue.svg?style=for-the-badge)
![Chroma_DB](https://img.shields.io/badge/Chroma%20DB-Optional-yellow.svg?style=for-the-badge)

> **TaxGPT** is a Streamlit-based chatbot that provides real-time Chain-of-Thought streaming for tax-related queries.  
> It uses LangChain’s RetrievalQA chain with an optional local PDF knowledge base to generate answers with internal reasoning.

---

## Key Features

- **Real-Time Chain-of-Thought Streaming**  
  The internal reasoning is streamed into a dropdown expander titled **"Chain of Thought"** (with `<think>` and `</think>` tags removed).

- **Real-Time Final Answer**  
  Tokens after the closing `</think>` tag are streamed as the final answer in real time.

- **Local PDF Knowledge Base (Optional)**  
  Provide a local **TAX.pdf** file in the project root. The app processes the PDF using PDFPlumberLoader and splits it into chunks for retrieval.

- **Clean Conversation History**  
  Only the final answer is stored in the conversation history to keep the chat concise.

---

## Tech Stack

- **[Python](https://www.python.org/)**  
- **[Streamlit](https://streamlit.io/)**  
- **[LangChain](https://github.com/hwchase17/langchain)**  
- **[PDFPlumberLoader](https://github.com/psychok7/langchain-community/blob/main/docs/loaders/pdf_plumber.md)**  
- **[Chroma DB](https://docs.trychroma.com/)** (Optional)

---

## Prerequisites

- Python 3.9+
- A local **TAX.pdf** file in the project root (optional for the knowledge base)
- Dependencies listed in `requirements.txt`

---

## Installation & Setup

1. **Clone the Repository**

    ```bash
    git clone https://github.com/your-username/taxgpt.git
    cd taxgpt
    ```

2. **(Optional) Create and Activate a Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate    # For Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **(Optional) Place your TAX.pdf in the Project Root**

---

## Running the App

Run the following command in your terminal:

```bash
streamlit run app.py
