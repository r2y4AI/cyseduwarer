import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama
import warnings

# Suppress torch warning
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional
from langchain.chains import RetrievalQA


# Set protobuf environment variable to avoid error messages
# This might cause some issues with latency but it's a tradeoff
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Define persistent directory for ChromaDB
PERSIST_DIRECTORY = os.path.join("data", "vectors")

# Streamlit page configuration
st.set_page_config(
    page_title="::CySEduWarer::",
    page_icon=":books:",
    layout="wide",
    # layout="centered",
    # initial_sidebar_state="collapsed", # auto
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def extract_model_names(models_info: Any) -> Tuple[str, ...]:
    """
    Extract model names from the provided models information.

    Args:
        models_info: Response from ollama.list()

    Returns:
        Tuple[str, ...]: A tuple of model names.
    """
    logger.info("Extracting model names from models_info")
    try:
        # The new response format returns a list of Model objects
        if hasattr(models_info, "models"):
            # Extract model names from the Model objects
            model_names = tuple(model.model for model in models_info.models)
        else:
            # Fallback for any other format
            model_names = tuple()
            
        logger.info(f"Extracted model names: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error extracting model names: {e}")
        return tuple()


def create_vector_db(file_upload) -> Chroma:
    """
    Create a vector database from an uploaded PDF file.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        Chroma: A vector store containing the processed document chunks.
    """
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")
        loader = UnstructuredPDFLoader(path)
        data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(data)
    logger.info("Document split into chunks")

    # Updated embeddings configuration with persistent storage
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=f"pdf_{hash(file_upload.name)}"  # Unique collection name per file
    )
    logger.info("Vector DB created with persistent storage")

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return vector_db


def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    """
    Process a user question using the vector database and selected language model.

    Args:
        question (str): The user's question.
        vector_db (Chroma): The vector database containing document embeddings.
        selected_model (str): The name of the selected language model.

    Returns:
        str: The generated response to the user's question.
    """
    logger.info(f"Processing question: {question} using model: {selected_model}")
    
    # Initialize LLM
    llm = ChatOllama(model=selected_model)

    # Query prompt template
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. 
        Your task is to generate 3 different versions of the given user question in information security and cybersecurity domain to retrieve relevant documents from a vector database. 
        By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    # Set up retriever
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), 
        llm,
        prompt=QUERY_PROMPT
    )

    context = retriever

    template = """
    You are a question-answering system for a self-learning of cybersecurity education and awareness. 
    Your task is to answer a given learner question for the control in ISO/IEC 27002:2022 standard.
    Follow up the next 4-shots examples, delimited by ####, to answer the learner question similary, clearly, and precisely. 

    ####
    Firt-Shot example: 

    Q: "Threat intelligence"
    A: "
    1) Attribute table:
        a) Control type: Preventive, Detective, and Corrective.
        b) Information security properties: Confidentiality, Integrity, and Availability.
        c) Cybersecurity concepts: Identify, Detect, and Respond.
        d) Operational capabilities: “Threat and vulnerability management”. 
        e) Security domains: Defence and Resilience.

    2) Control: Information relating to information security threats should be collected and analyzed to produce threat intelligence.

    3) Purpose: To provide awareness of the organization threat environment so that the appropriate mitigation actions can be taken.

    4) Guidance: Information about existing or emerging threats is collected and analyzed in order to:
        a) facilitate informed actions to prevent the threats from causing harm to the organization;
        b) reduce the impact of such threats.
    
    Threat intelligence can be divided into three layers, which should all be considered:
        a) strategic threat intelligence: exchange of high-level information about the changing threat landscape (e.g. types of attackers or types of attacks);
        b) tactical threat intelligence: information about attacker methodologies, tools and technologies involved;
        c) operational threat intelligence: details about specific attacks, including technical indicators. Threat intelligence should be:
            a) relevant (i.e. related to the protection of the organization);
            b) insightful (i.e. providing the organization with an accurate and detailed understanding of the threat landscape);
            c) contextual, to provide situational awareness (i.e. adding context to the information based on the time of events, where they occur, previous experiences and prevalence in similar organizations);
            d) actionable (i.e. the organization can act on information quickly and effectively). Threat intelligence activities should include:
                a) establishing objectives for threat intelligence production;
                b) identifying, vetting and selecting internal and external information sources that are necessary and appropriate to provide information required for the production of threat intelligence;
                c) collecting information from selected sources, which can be internal and external;
                d) processing information collected to prepare it for analysis (e.g. by translating, formatting or corroborating information);
                e) analyzing information to understand how it relates and is meaningful to the organization;
                f) communicating and sharing it to relevant individuals in a format that can be understood. Threat intelligence should be analyzed and later used:
                    a) by implementing processes to include information gathered from threat intelligence sources into the organization information security risk management processes;
                    b) as additional input to technical preventive and detective controls like firewalls, intrusion detection system, or anti malware solutions;
                    c) as input to the information security test processes and techniques.
                
    The organization should share threat intelligence with other organizations on a mutual basis in order to improve overall threat intelligence."
    
    ####

    ####
    Second-Shot example: 

    Q: "awareness, education and training"
    A: "

    1) Attribute table:
        a) Control type: Preventive.
        b) Information security properties: Confidentiality, Integrity, and Availability.
        c) Cybersecurity concepts: Protect.
        d) Operational capabilities: “Human resource security”. 
        e) Security domains: Governance and Ecosystem.

    2) Control: Personnel of the organization and relevant interested parties should receive appropriate information security awareness, education and training and regular updates of the organization's information security policy, topic-specific policies and procedures, as relevant for their job function.
    3) Purpose: To ensure personnel and relevant interested parties are aware of and fulfil their information security responsibilities.
    4) Guidance:
        General
        An information security awareness, education and training program should be established in line with the organization information security policy, topic-specific policies and relevant procedures on information security, taking into consideration the organization information to be protected and the information security controls that have been implemented to protect the information.
        Information security awareness, education and training should take place periodically. Initial awareness, education and training can apply to new personnel and to those who transfer to new positions or roles with substantially different information security requirements.
        Personnel understanding should be assessed at the end of an awareness, education or training activity to test knowledge transfer and the effectiveness of the awareness, education and training program.
        Awareness
        An information security awareness program should aim to make personnel aware of their responsibilities for information security and the means by which those responsibilities are discharged.
        The awareness program should be planned taking into consideration the roles of personnel in the organization, including internal and external personnel (e.g. external consultants, supplier personnel). The activities in the awareness program should be scheduled over time, preferably regularly, so that the activities are repeated and cover new personnel. It should also be built on lessons learnt from information security incidents.
        The awareness program should include a number of awareness-raising activities via appropriate physical or virtual channels such as campaigns, booklets, posters, newsletters, websites, information sessions, briefings, e-learning modules and e-mails.
        Information security awareness should cover general aspects such as:
            a) management commitment to information security throughout the organization;
            b) familiarity and compliance needs concerning applicable information security rules and obligations, taking into account information security policy and topic-specific policies, standards, laws, statutes, regulations, contracts and agreements;
            c) personal accountability for one own actions and inactions, and general responsibilities towards securing or protecting information belonging to the organization and interested parties;
            d) basic information security procedures [e.g. information security event reporting (6.8)] and baseline controls [e.g. password security (5.17)];
            e) contact points and resources for additional information and advice on information security matters, including further information security awareness materials.
        Education and training
    The organization should identify, prepare and implement an appropriate training plan for technical teams whose roles require specific skill sets and expertise. Technical teams should have the skills for configuring and maintaining the required security level for devices, systems, applications and services. If there are missing skills, the organization should take action and acquire them.
    The education and training program should consider different forms [e.g. lectures or self-studies, being mentored by expert staff or consultants (on-the-job training), rotating staff members to follow different activities, recruiting already skilled people and hiring consultants]. It can use different means of delivery including classroom-based, distance learning, web-based, self-paced and others. Technical personnel should keep their knowledge up to date by subscribing to newsletters and magazines or by attending conferences and events aimed at technical and professional improvement."
    ####

    ####
    
    Third-Shot example: 

    Q: "Equipment siting and protection"
    A: "

    1) Attribute table:
        a) Control type: Preventive.
        b) Information security properties: Confidentiality, Integrity, and Availability.
        c) Cybersecurity concepts: Protect.
        d) Operational capabilities: “Physical security” and “Asset management”. 
        e) Security domains: Protection.

    2) Control: Equipment should be sited securely and protected.

    3) Purpose: To reduce the risks from physical and environmental threats, and from unauthorized access and damage.

    4) Guidance:
        The following guidelines should be considered to protect equipment:
            a) siting equipment to minimize unnecessary access into work areas and to avoid unauthorized access;
            b) carefully positioning information processing facilities handling sensitive data to reduce the risk of information being viewed by unauthorized persons during their use;
            c) adopting controls to minimize the risk of potential physical and environmental threats [e.g. theft, fire, explosives, smoke, water (or water supply failure), dust, vibration, chemical effects, electrical supply interference, communications interference, electromagnetic radiation and vandalism];
            d) establishing guidelines for eating, drinking and smoking in proximity to information processing facilities;
            e) monitoring environmental conditions, such as temperature and humidity, for conditions which can adversely affect the operation of information processing facilities;
            f) applying lightning protection to all buildings and fitting lightning protection filters to all incoming power and communications lines;
            g) considering the use of special protection methods, such as keyboard membranes, for equipment in industrial environments;
            h) protecting equipment processing confidential information to minimize the risk of information leakage due to electromagnetic emanation;
            i) physically separating information processing facilities managed by the organization from those not managed by the organization."
    
    ####

    Fourth-Shot example: 

    Q: "Capacity management"
    A: "

    1) Attribute table:
        a) Control type: Preventive and Detective.
        b) Information security properties: Integrity and Availability.
        c) Cybersecurity concepts: Identify, Detect, and Detect.
        d) Operational capabilities: Continuity. 
        e) Security domains: “Governance and Ecosystem” and “Protection”.

    2) Control: The use of resources should be monitored and adjusted in line with current and expected capacity requirements.

    3) Purpose: To ensure the required capacity of information processing facilities, human resources, offices and other facilities.

    4) Guidance:
        Capacity requirements for information processing facilities, human resources, offices and other facilities should be identified, taking into account the business criticality of the concerned systems and processes.
        System tuning and monitoring should be applied to ensure and, where necessary, improve the availability and efficiency of systems.
        The organization should perform stress-tests of systems and services to confirm that sufficient system capacity is available to meet peak performance requirements.
        Detective controls should be put in place to indicate problems in due time.
        Projections of future capacity requirements should take account of new business and system requirements and current and projected trends in the organization information processing capabilities.
        Particular attention should be paid to any resources with long procurement lead times or high costs. Therefore, managers, service or product owners should monitor the utilization of key system resources.
        Managers should use capacity information to identify and avoid potential resource limitations and dependency on key personnel which can present a threat to system security or services and plan appropriate action.
        Providing sufficient capacity can be achieved by increasing capacity or by reducing demand. The following should be considered to increase capacity:
            a) hiring new personnel;
            b) obtaining new facilities or space;
            c) acquiring more powerful processing systems, memory and storage;
            d) making use of cloud computing, which has inherent characteristics that directly address issues of capacity. Cloud computing has elasticity and scalability which enable on-demand rapid expansion and reduction in resources available to particular applications and services.

    The following should be considered to reduce demand on the organization resources:
        a) deletion of obsolete data (disk space);
        b) disposal of hardcopy records that have met their retention period (free up shelving space);
        c) decommissioning of applications, systems, databases or environments;
        d) optimizing batch processes and schedules;
        e) optimizing application code or database queries;
        f) denying or restricting bandwidth for resource-consuming services if these are not critical (e.g. video streaming).
    A documented capacity management plan should be considered for mission critical systems."
    
    ####


    Question: {question}
    Context:  {context}
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}, # , "context": context
        return_source_documents=True,
        verbose=True
    )

    response = chain.invoke(question)
    logger.info("Question processed and response generated")
    return response


@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """
    Extract all pages from a PDF file as images.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        List[Any]: A list of image objects representing each page of the PDF.
    """
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages


def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """
    Delete the vector database and clear related session state.

    Args:
        vector_db (Optional[Chroma]): The vector database to be deleted.
    """
    logger.info("Deleting vector DB")
    if vector_db is not None:
        try:
            # Delete the collection
            vector_db.delete_collection()
            
            # Clear session state
            st.session_state.pop("pdf_pages", None)
            st.session_state.pop("file_upload", None)
            st.session_state.pop("vector_db", None)
            
            st.success("Collection and temporary files deleted successfully.")
            logger.info("Vector DB and related session state cleared")
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting collection: {str(e)}")
            logger.error(f"Error deleting collection: {e}")
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")


def main() -> None:
    st.header(":books: :red[CyS]:blue[EduWarer:]")
    st.subheader("Q&A System for Self-Learning of Cybersecurity Education and Awareness Based on LLMs Using RAG", divider="gray", anchor=False)
    st.markdown(":grey[Mohammed A. Saleh (m.saleh@qu.edu.sa)]")
    # st.markdown("#")
    # st.header(divider="gray")

    # Get available models
    models_info = ollama.list()
    available_models = extract_model_names(models_info)

    # Create layout
    col1, col2 = st.columns([1, 3])
    # col1, col2 = st.columns([1.5, 2])

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None
    if "use_sample" not in st.session_state:
        st.session_state["use_sample"] = False


    # Add checkbox for sample PDF
    use_sample = col1.toggle(
        "Use ISO/IEC 27002:2022", 
        key="sample_checkbox"
    )

    
    # Clear vector DB if switching between sample and upload
    if use_sample != st.session_state.get("use_sample"):
        if st.session_state["vector_db"] is not None:
            st.session_state["vector_db"].delete_collection()
            st.session_state["vector_db"] = None
            st.session_state["pdf_pages"] = None
        st.session_state["use_sample"] = use_sample

    if use_sample:
        # Use the sample PDF
        sample_path = "data/pdfs/sample/ISO_IEC 27002-2022.pdf"
        if os.path.exists(sample_path):
            if st.session_state["vector_db"] is None:
                with st.spinner("Processing ..."):
                    loader = UnstructuredPDFLoader(file_path=sample_path)
                    data = loader.load()
                    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = text_splitter.split_documents(data)
                    st.session_state["vector_db"] = Chroma.from_documents(
                        documents=chunks,
                        embedding=OllamaEmbeddings(model="nomic-embed-text"),
                        persist_directory=PERSIST_DIRECTORY,
                        collection_name="sample_pdf"
                    )
                    # Open and display the sample PDF
                    with pdfplumber.open(sample_path) as pdf:
                        st.session_state["pdf_pages"] = [page.to_image().original for page in pdf.pages]
        else:
            st.error("Security/Cybersecurity Standards PDF File not found in the current directory.")
    else:
        # Regular file upload with unique key
        file_upload = col1.file_uploader(
            "Upload Security/Cybersecurity Standards PDF File ↓", 
            type="pdf", 
            accept_multiple_files=False,
            key="pdf_uploader"
        )

        if file_upload:
            if st.session_state["vector_db"] is None:
                with st.spinner("Processing uploaded PDF..."):
                    st.session_state["vector_db"] = create_vector_db(file_upload)
                    # Store the uploaded file in session state
                    st.session_state["file_upload"] = file_upload
                    # Extract and store PDF pages
                    with pdfplumber.open(file_upload) as pdf:
                        st.session_state["pdf_pages"] = [page.to_image().original for page in pdf.pages]

    # Model selection
    if available_models:
        # selected_model = col2.selectbox(
        selected_model = col1.selectbox(
            "Select Local LLM Model ↓", 
            available_models,
            key="model_select"
        )

    # Delete collection button
    delete_collection = col1.button(
        "⚠️ Delete Knowledge DB", 
        type="secondary",
        key="delete_button"
    )

    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])

    # Chat interface
    with col2:
        # message_container = st.container(height=500, border=True)
        greeting = "Hello, i am CySEduWarer. How can I assist you?"
        st.subheader(greeting)

        user_question = st.chat_input("Ask a question about Cybersecurity Education and Awareness ...", key="chat_input")

        message_container = st.container(height=500, border=True)

        # Display chat history
        for i, message in enumerate(st.session_state["messages"]):
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            # with message_container.chat_message(message["role"], avatar=avatar):
            with message_container.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input and processing
        # if prompt := st.chat_input("Ask a question about Cybersecurity Education and Awareness ...", key="chat_input"):
        if prompt := user_question:
            try:
                # message_container = st.container(height=500, border=True)
                # Add user message to chat
                st.session_state["messages"].append({"role": "user", "content": prompt})
                # with message_container.chat_message("user", avatar="😎"):
                with message_container.chat_message("user"):
                    st.markdown(":red[Your Question: ]")
                    st.markdown(prompt)

                # Process and display assistant response
                # with message_container.chat_message("assistant", avatar="🤖"):
                with message_container.chat_message("assistant"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            response = process_question(
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            st.markdown(":blue[CySEduWarer Answer: ]")
                            st.markdown(response["result"])
                            # printout RAG context
                            st.markdown(":red[RAG Context: ]")
                            # st.markdown(response["source_documents"])
                            for i, doc in enumerate(response["source_documents"], 1):
                                st.markdown(doc.page_content)
                        else:
                            st.markdown("CySEduWarer Answer: ")
                            st.warning("Please upload a Security/Cybersecurity Standards PDF file first.")

                # Add assistant response to chat history
                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )

            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Hint: Upload Security/Cybersecurity Standards PDF File.")


if __name__ == "__main__":
    main()