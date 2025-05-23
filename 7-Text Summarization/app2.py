import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader



st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')


with st.sidebar:
    groq_api_key = st.text_input("Groq_api_key",value = "",type = "password")

generic_url = st.text_input("URL",label_visibility="collapsed")

llm = ChatGroq(groq_api_key = groq_api_key,model_name = "meta-llama/llama-4-scout-17b-16e-instruct")

prompt = PromptTemplate(
    template="""Provide the summary of the following context in 300 words:
Context:{text}""",
    input_variables=["text"]
)



if st.button("summarize the content from YT or website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Provide the information")
    elif not validators.url(generic_url):
        st.error("please enter valid url. It can may be YT video utl or website url")
        
    else:
        try:
            with st.spinner("Waiting..."):
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url],ssl_verify = False,
                                                   headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    docs = loader.load()
                    
                    
                    chain = load_summarize_chain(llm,chain_type = "stuff",prompt = prompt)
                    output_summary = chain.run(docs)
                    
                    st.success(output_summary)
        except Exception as e:
             st.exception(f"Exception:{e}")
            
                
                    
                    
                    
                    
            


    

