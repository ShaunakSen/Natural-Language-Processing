import os
from pprint import pprint
import logging
import tempfile

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import AzureChatOpenAI
from langchain.llms import AzureOpenAI

import chainlit as cl
from chainlit.types import AskFileResponse

from creds import AZURE_API_BASE, AZURE_API_VERSION, AZURE_API_KEY

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CHATBOT_NAME = "miniGPT"

def init_models() -> list:
    embeddings = OpenAIEmbeddings(
        deployment="text-embedding-ada-002",
        model="text-embedding-ada-002",
        openai_api_type='azure',
        openai_api_base=AZURE_API_BASE,
        openai_api_key=AZURE_API_KEY,
        openai_api_version=AZURE_API_VERSION,
        chunk_size=1, max_retries=1e0 
    )

    llm_chat_gpt_4 = AzureChatOpenAI(deployment_name='gpt-4-32k',
                          model='gpt-4-32k',
                          openai_api_type='azure',
                          openai_api_base=AZURE_API_BASE,
                          openai_api_key=AZURE_API_KEY,
                          openai_api_version=AZURE_API_VERSION,
                          max_retries=2,
                          temperature=0,
                          streaming=True
                          )

    llm_gpt_4 = AzureOpenAI(deployment_name='gpt-4-32k',
                            model='gpt-4-32k',
                            openai_api_type='azure',
                            openai_api_base=AZURE_API_BASE,
                            openai_api_key=AZURE_API_KEY,
                            openai_api_version=AZURE_API_VERSION,
                            max_retries=2,
                            temperature=0,
                            streaming=True
                            )
    
    return [embeddings, llm_chat_gpt_4]

def setup_text_splitter(chunk_size: int=1000, chunk_overlap: int=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter


def process_file(file: AskFileResponse) -> list:
    """
    Returns a list of documents
    """

    if file.type == "text/plain":
        loader = TextLoader
    elif file.type == "application/pdf":
        loader = PyPDFLoader
    
    with tempfile.NamedTemporaryFile() as temp:
        temp.write(file.content)
        logger.debug(temp.name)

        documents = loader(temp.name).load()
        logger.debug(f"Length of docs: {len(documents)}")
        text_splitter = setup_text_splitter()
        split_docs = text_splitter.split_documents(documents)
        for doc_ in split_docs:
            doc_.metadata['source'] = file.name
        logger.debug(split_docs)

        return split_docs
    
def build_retriever(docs: list, embeddings):
    # save docs in user session
    cl.user_session.set("docs", docs)

    docsearch = Chroma.from_documents(documents=docs, embedding=embeddings)

    return docsearch


def build_chain(docsearch, chat_model):
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=chat_model,
        chain_type="stuff",
        verbose=True,
        retriever=docsearch.as_retriever(max_tokens_limit=4097)
    )

    return chain





@cl.on_chat_start
async def start():
    """
    """
    # TODO: take input for a tone of a popular character and send the responses in that tone
    # send a message
    msg = cl.Message(content="Chatbot to chat with local pdf files", author=CHATBOT_NAME)
    await msg.send()

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Upload your file (pdf)",
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180
        ).send()

    file = files[0]

    logger.debug((file.name, file.path, file.type))

    await cl.Message(content=f"Processing {file.name}...").send()

    embeddings, chat_llm = await cl.make_async(init_models)()
    documents = await cl.make_async(process_file)(file)
    docsearch = await cl.make_async(build_retriever)(documents, embeddings)

    chain = await cl.make_async(build_chain)(docsearch, chat_llm)

    # Let the user know that the system is ready
    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: str):

    # retrieve the chain stored in user session
    chain = cl.user_session.get("chain")
    callback = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    callback.answer_reached = True
    res = await chain.acall(message, callbacks=[callback])

    pprint(res)


    





    



