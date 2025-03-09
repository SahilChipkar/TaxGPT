import streamlit as st
import tempfile
import os, re
from io import BytesIO

from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager

def process_documents(pdf_files):
    """
    Process the provided PDF BytesIO objects by writing them to a temporary file,
    loading them with PDFPlumberLoader, and splitting into chunks.
    """
    from langchain_community.document_loaders import PDFPlumberLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    documents = []
    for pdf_file in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name

        loader = PDFPlumberLoader(tmp_path)
        documents.extend(loader.load())
        os.remove(tmp_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    splits = text_splitter.split_documents(documents)
    return splits

def load_vector_store():
    """
    Loads the existing vector store from the persist directory.
    """
    from langchain_ollama import OllamaEmbeddings
    from langchain_chroma import Chroma
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

def load_tax_pdf_file():
    """
    Returns a BytesIO object for TAX.pdf if it exists, else None.
    """
    pdf_path = "./TAX.pdf"
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            return BytesIO(f.read())
    return None

class StreamBothCallbackHandler(BaseCallbackHandler):
    """
    Streams tokens into two placeholders:
      - Tokens up to and including </think> are accumulated as chain-of-thought.
      - Tokens after </think> are accumulated as the final answer.
    When displaying the chain-of-thought, the <think> and </think> tags are removed.
    """
    def __init__(self, cot_placeholder, answer_placeholder):
        super().__init__()
        self.cot_placeholder = cot_placeholder
        self.answer_placeholder = answer_placeholder
        self.full_text = ""
        self.seen_close_tag = False
        self.cot_text = ""
        self.answer_text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        # As tokens arrive, direct them to either chain-of-thought or final answer
        self.full_text += token
        if not self.seen_close_tag:
            idx_close = token.find("</think>")
            if idx_close >= 0:
                # Up to idx_close is chain-of-thought
                self.cot_text += token[: idx_close + len("</think>")]
                # Remove <think> and </think> tags for display
                clean_cot = re.sub(r"</?think>", "", self.cot_text).strip()
                self.cot_placeholder.markdown(clean_cot)
                self.seen_close_tag = True

                # Remainder is final answer
                remainder = token[idx_close + len("</think>"):]
                self.answer_text += remainder
                self.answer_placeholder.markdown(f"**Final Answer:**\n\n{self.answer_text}")
            else:
                self.cot_text += token
                clean_cot = re.sub(r"</?think>", "", self.cot_text).strip()
                self.cot_placeholder.markdown(clean_cot)
        else:
            # We are in final answer territory
            self.answer_text += token
            self.answer_placeholder.markdown(f"**Final Answer:**\n\n{self.answer_text}")

def get_custom_prompt():
    """
    Instructs the LLM to enclose its internal reasoning between <think> and </think>
    tags, then provide the final answer after the closing tag.
    """
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an AI assistant specialized in the new tax regime. Your responses must rely exclusively on the provided tax regime information and reference specific clauses when relevant.\n\n"
            "Guidelines:\n"
            "1. Answer questions using only the provided tax regime information. Do not incorporate external knowledge.\n"
            "2. For salary income deductions, refer explicitly to the provisions in Clause 19 – 'Deductions from Salaries'. For other topics, cite the relevant clauses.\n"
            "3. Use clear, concise language that is practical for real-life tax scenarios.\n"
            "4. If the required information is not found, respond with: 'I cannot find relevant information in the provided tax regime.'\n"
            "5. Do not speculate or invent details—stick strictly to the provided information.\n"
            "6. Organize your answer logically and provide examples where helpful.\n"
            "7. Enclose your internal chain-of-thought reasoning within <think> and </think> tags. Everything before </think> is your reasoning; everything after is the final answer.\n\n"
            "Example:\n<think>This is my hidden reasoning.</think>Real Answer: [final answer here]"
        ),
        HumanMessagePromptTemplate.from_template(
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Provide your answer following the chain-of-thought format."
        )
    ])

def initialize_qa_chain(callback_manager=None):
    st.session_state.qa_chain = None
    if st.session_state.vector_store:
        llm = ChatOllama(model="deepseek-r1", temperature=0.3, stream=True, callback_manager=callback_manager)
        retriever = st.session_state.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": get_custom_prompt()}
        )
    return st.session_state.qa_chain

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        if os.path.exists("./chroma_db") and os.listdir("./chroma_db"):
            st.session_state.vector_store = load_vector_store()
        else:
            st.session_state.vector_store = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

def create_knowledge_base():
    if not st.session_state.vector_store:
        pdf_file = load_tax_pdf_file()
        if pdf_file:
            with st.spinner("Processing tax regime information..."):
                try:
                    doc_splits = process_documents([pdf_file])
                    # Typically, you'd embed these documents to build a vector store.
                    # For demonstration, we'll just set vector_store to None.
                    st.session_state.vector_store = None
                    st.session_state.qa_chain = None
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
        else:
            st.error("Tax regime document not found in the current directory.")

def chat_interface():
    # Rename the site to "TaxGPT"
    st.title("TaxGPT")
    st.markdown("Your AI assistant for the new tax regime.")
    
    with st.sidebar:
        st.markdown("### Topics You Can Ask About")
        st.markdown(
            """
            - **Salary Income Deductions:** Provident Fund, Pension, Gratuity, etc.
            - **Long-Term Capital Gains:** Indexation benefits, capital gains computation.
            - **Business Income Deductions:** Deductions from business income, depreciation, etc.
            - **Other Income Taxation:** Dividends, interest, royalties, etc.
            - **Exemptions and Allowances:** Residential property exemptions, special deductions.
            - **Tax on Foreign Transactions:** Taxation on foreign exchange gains, etc.
            - **Advance Tax Calculations:** How to compute and pay advance tax.
            - **Loss Set-Off and Carry Forward:** Rules for carrying forward losses.
            - **Collection and Recovery:** Procedures for tax collection and recovery.
            - **General Anti-Avoidance Rules:** Provisions to prevent tax avoidance.
            - **Appeals and Revisions:** Procedures for appeals, revisions, and dispute resolution.
            """
        )

    create_knowledge_base()

    # Display conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask a tax-related question"):
        greetings = {"hi", "hello", "hey", "hi there", "greetings"}
        if prompt.strip().lower() in greetings:
            bot_reply = "Hello! How can I help you with your tax-related queries?"
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            with st.chat_message("assistant"):
                st.markdown(bot_reply)
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                # Use "Chain of Thought" as the expander label
                cot_expander = st.expander("Chain of Thought", expanded=True)
                cot_placeholder = cot_expander.empty()
                answer_placeholder = st.empty()

                with st.spinner("Thinking..."):
                    try:
                        callback = StreamBothCallbackHandler(cot_placeholder, answer_placeholder)
                        cb_manager = CallbackManager([callback])
                        qa_chain = initialize_qa_chain(callback_manager=cb_manager)
                        if not qa_chain:
                            final_answer = "Knowledge base not created. Possibly no PDF or no vector store."
                        else:
                            _ = qa_chain.invoke({"query": prompt})
                            # final answer is the text after streaming is done
                            final_answer = callback.answer_text
                    except Exception as e:
                        final_answer = f"Error: {str(e)}"
            
            # Store only the final answer in conversation history (avoid duplication)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

def main():
    initialize_session_state()
    chat_interface()

if __name__ == "__main__":
    main()
