
# pip install -U langchain-openai
# switch back to cmd -  venvlangchain\Scripts\activate
# pip install -r requirements.txt 
# To run in the cmd: streamlit run gpt.py


import streamlit as st
#from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
#from langchain import OpenAI
#from langchain_community.llms import OpenAI
#from langchain_community.chat_models import ChatOpenAI
#from langchain import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter


import os
from constants import openai_key
os.environ["OPENAI_API_KEY"] = openai_key
import tempfile

llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

# Function to summarize PDFs from a folder
def summarize_pdfs_from_folder(pdfs_folder):
    summaries = []
    for pdf_file in pdfs_folder:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(pdf_file.read())
        
        pdfreader = PdfReader(temp_path)
        text = ''
        for i, page in enumerate(pdfreader.pages):
            content = page.extract_text()
            if content:
                text += content

        llm.get_num_tokens(text)

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
        chunks = text_splitter.create_documents([text])

        # Define prompts for summarization
        chunks_prompt = """
        Please summarize the following financial document segment:
        Segment:`{text}`
        Summary:
        """
        map_prompt_template = PromptTemplate(input_variables=['text'], template=chunks_prompt)

        final_combine_prompt = '''
        Provide a final summary of the entire financial document with these important points.
        Add a title summarizing the key findings,
        Start the precise summary with an introduction and provide the
        summary in numbered points, highlighting the main sections and insights of the document.
        Document: `{text}`
        '''
        final_combine_prompt_template = PromptTemplate(input_variables=['text'], template=final_combine_prompt)

        # Load and run the summarization chain
        summary_chain = load_summarize_chain(llm=llm, chain_type='map_reduce',
                                             map_prompt=map_prompt_template, combine_prompt=final_combine_prompt_template,
                                             verbose=False)
        final_summary = summary_chain.invoke(chunks)
        summaries.append(final_summary)

        # Delete the temporary file
        os.remove(temp_path)
    
    return summaries

# Streamlit App
st.title("Document Summarizer with Custom Prompt")

# Allow user to upload PDF files
pdf_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Generate summaries when the "Generate Summaries" button is clicked
if st.button("Generate Summaries") and pdf_files:
    summaries = summarize_pdfs_from_folder(pdf_files)
    for i, summary in enumerate(summaries):
        st.write(f"Summary for PDF {i+1}:")
        st.write(summary)
