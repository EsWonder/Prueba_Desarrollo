import os
import streamlit as st
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import time  # Importar el módulo time

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = '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'


def process_html(html_file):
    html_text = html_file.read()
    soup = BeautifulSoup(html_text, "html.parser")
    text = soup.get_text()
    return text


def process_xml(xml_file):
    xml_text = xml_file.read()
    root = ET.fromstring(xml_text)
    entries_text = [element.text if element.text is not None else '' for element in root.iter()]
    text = ' '.join(entries_text)
    return text


def generate_response(uploaded_file, query_text, file_type):
    start_time = time.time()  # Registrar el tiempo de inicio
    if file_type == 'html':
        processed_text = process_html(uploaded_file)
    elif file_type == 'xml':
        processed_text = process_xml(uploaded_file)
    else:
        st.error(f"Tipo de archivo no admitido: {file_type}")
        return None, None, None

    # Reduce la longitud del texto si es necesario
    max_context_length = 4097
    processed_text = processed_text[:max_context_length]

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents([processed_text])

    # Select embeddings
    embeddings = OpenAIEmbeddings()

    # Create a vectorstore from documents
    db = Chroma.from_documents(texts, embeddings)

    # Create retriever interface
    retriever = db.as_retriever()

    # Create QA chain
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type='stuff', retriever=retriever)
    response = qa.run(query_text)

    elapsed_time = time.time() - start_time  # Calcular el tiempo transcurrido
    tokens_consumed = len(response.split()) if response else 0  # Contar los tokens consumidos

    return response, elapsed_time, tokens_consumed


# Page title
st.set_page_config(page_title='💻🔗 Prueba Desarrollo')
st.title('💻🔗 Uso de Chroma por Faiss')

# File upload HTML
uploaded_html_file = st.file_uploader('Sube un artículo HTML', type=['html', 'htm'])
query_html_text = st.text_input('Ingresa tu pregunta para HTML:', placeholder='Por favor proporciona un breve resumen.',
                                disabled=not uploaded_html_file)

# File upload XML
uploaded_xml_file = st.file_uploader('Sube un artículo XML', type=['xml'])
query_xml_text = st.text_input('Ingresa tu pregunta para XML:', placeholder='Por favor proporciona un breve resumen.',
                               disabled=not uploaded_xml_file)

# Form input and query for HTML
result_html = []
with st.form('html_form', clear_on_submit=True):
    submitted_html = st.form_submit_button('Enviar para HTML', disabled=not (uploaded_html_file and query_html_text))
    if submitted_html:
        with st.spinner('Calculando...'):
            response_html, elapsed_time_html, tokens_consumed_html = generate_response(uploaded_html_file,
                                                                                       query_html_text, 'html')
            result_html.append(response_html)

if len(result_html):
    st.info(f"Respuesta HTML: {response_html}")
    st.info(f"Tiempo transcurrido HTML: {elapsed_time_html} segundos")
    st.info(f"Tokens consumidos HTML: {tokens_consumed_html}")

# Form input and query for XML
result_xml = []
with st.form('xml_form', clear_on_submit=True):
    submitted_xml = st.form_submit_button('Enviar para XML', disabled=not (uploaded_xml_file and query_xml_text))
    if submitted_xml:
        with st.spinner('Calculando...'):
            response_xml, elapsed_time_xml, tokens_consumed_xml = generate_response(uploaded_xml_file, query_xml_text,
                                                                                    'xml')
            result_xml.append(response_xml)

if len(result_xml):
    st.info(f"Respuesta XML: {response_xml}")
    st.info(f"Tiempo transcurrido XML: {elapsed_time_xml} segundos")
    st.info(f"Tokens consumidos XML: {tokens_consumed_xml}")
