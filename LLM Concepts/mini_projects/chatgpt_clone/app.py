from pprint import pprint
from langchain import PromptTemplate
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
import chainlit as cl

from creds import AZURE_API_BASE, AZURE_API_KEY, AZURE_API_VERSION

template = """Question: {question}

Answer: Let's think step by step."""

def init_llm():
    llm_gpt_4 = AzureChatOpenAI(deployment_name='gpt-4-32k',
                          model='gpt-4-32k',
                          openai_api_type='azure',
                          openai_api_base=AZURE_API_BASE,
                          openai_api_key=AZURE_API_KEY,
                          openai_api_version=AZURE_API_VERSION,
                          max_retries=2,
                          temperature=0,
                          streaming=True
                          )
    
    return llm_gpt_4

@cl.on_chat_start # as soon as the cl UI is up, this runs
def main():
    llm = init_llm()
    # Instantiate the chain for that user session
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain asynchronously
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Do any post processing here

    # "res" is a Dict. For this chain, we get the response by reading the "text" key.
    # This varies from chain to chain, you should check which key to read.

    pprint (res)

    await cl.Message(content=res["text"]).send()

