import streamlit as st
import asyncio
import time
import os
from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient, AzureOpenAIChatCompletionClient
from autogen_ext.teams.magentic_one import MagenticOne
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from datetime import date, timedelta


load_dotenv()

def format_source_display(source):
    """
    Converts a source identifier into a user-friendly display string with an appropriate emoji.
    
    Args:
        source (str): The message source identifier
        
    Returns:
        str: Formatted string with emoji representing the source
    """
    if source == "user":
        return "👤 User"
    elif source == "MagenticOneOrchestrator":
        return "🤖 MagenticOneOrchestrator"
    elif source == "WebSurfer":
        return "🌐 WebSurfer"
    elif source == "FileSurfer":
        return "📁 FileSurfer"
    elif source == "Coder":
        return "💻 Coder"
    else:
        return f"💻 Terminal"

async def run_task(user_prompt: str, USE_AOAI, model_name=None):
    """
    Executes a task with the given user prompt using either Azure OpenAI or OpenAI.
    Streams and displays results in the Streamlit UI as they become available.
    
    Args:
        user_prompt (str): The task prompt from the user
        USE_AOAI (bool): Whether to use Azure OpenAI API
        
    Yields:
        Various message chunks and task results
    """
    start_time = time.time()
    if(USE_AOAI):
        client = AzureOpenAIChatCompletionClient(
            azure_endpoint=os.getenv('AZURE_OPEN_AI_ENDPOINT'),
            model=model_name,
            api_version="2024-12-01-preview",
            api_key=os.getenv('AZURE_OPEN_AI_KEY')
        )
    else:
        client = OpenAIChatCompletionClient(
            model=os.getenv('OPEN_AI_MODEL_NAME'),
            api_key=os.getenv('OPEN_AI_API_KEY')
        )

    m1 = MagenticOne(client=client, code_executor=LocalCommandLineCodeExecutor())
    async for chunk in m1.run_stream(task=user_prompt):
        if chunk.__class__.__name__ != 'TaskResult':
            st.write(f"**{format_source_display(chunk.source)}**")
            if chunk.type == 'MultiModalMessage':
                image = 'data:image/png;base64,' + chunk.content[1].to_base64()
                st.image(image)
            else:
                st.markdown(chunk.content)
        else:
            st.write(f"**Task completed in {(time.time() - start_time):.2f} s.**")
        yield chunk
    yield None, time.time() - start_time

async def collect_results(user_prompt: str, USE_AOAI, model_name=None):
    """
    Collects all results from run_task and accumulates token usage statistics.
    Updates session state with token counts.
    
    Args:
        user_prompt (str): The task prompt from the user
        USE_AOAI (bool): Whether to use Azure OpenAI API
        
    Returns:
        list: Collection of all result chunks
    """
    results = []
    async for chunk in run_task(user_prompt, USE_AOAI, model_name):
        results.append(chunk)
    for result in results:
        if result is not None and result.__class__.__name__ == 'TaskResult':
            print(result)
            for message in result.messages:
                if message.source != "user":
                    if message.models_usage:
                        st.session_state.prompt_token = message.models_usage.prompt_tokens + st.session_state.prompt_token
                        st.session_state.completion_token = message.models_usage.completion_tokens + st.session_state.completion_token
    return results

def main():
    st.title('🧠🤖 AI Flight Ticket Finder')
    st.write("Get an AI Agent to search for the best flight ticket prices on PriceBreaker! Just enter your travel dates and airports below and let our smart assistant do the heavy lifting.")

    st.sidebar.title('Settings')
    USE_AOAI = st.sidebar.checkbox("Use Azure OpenAI", value=True)

    if(USE_AOAI):
        aoai_model_options = ["gpt-4o", "gpt-4o-mini", "o3-mini"]
        selected_model = st.sidebar.selectbox("Select Model", aoai_model_options, index=2)

    if 'output' not in st.session_state:
        st.session_state.output = None
        st.session_state.elapsed = None
        st.session_state.prompt_token = 0
        st.session_state.completion_token = 0

    # prompt = st.text_input('What is the task today?', value='')
    prompt = ""

    st.markdown("---")
    st.subheader("Compare Flight Ticket Prices from PriceBreaker")
    departure_date = st.date_input('Departure Date', value=date.today() + timedelta(days=7))
    return_date = st.date_input('Return Date', value=date.today() + timedelta(days=8))
    departure_airport = st.text_input('Departure Airport', value='SIN')
    return_airport = st.text_input('Return Airport', value='KUL')

    # New Input Variables
    no_of_pax = st.number_input("No. of Pax", min_value=1, value=1, step=1)
    preferred_airline = st.text_input("Preferred Airline", value="")
    travel_class = st.selectbox("Class", options=["Economy", "Premium Economy", "Business", "First"])

    if st.button("Run"):
        customized_prompt = f"""
        Here's a sample of pricebreaker.travel website url, read the url query and make changes accordingly.
        Sample url: "https://www.pricebreaker.travel/flights?source=webconnect&od1.origin_airport.code=SIN&od1.origin_datetime=2026-03-08&od2.origin_datetime=2026-03-11&ptc_adt=1&ptc_cnn=0&ptc_inf=0&cabin=Y&od2.origin_airport.code=ICN"
        Now, help me craft the full url where the travel date is from {departure_date} to {return_date}, from {departure_airport} to {return_airport}. 
        Passengers: {no_of_pax}. Preferred Airline: {preferred_airline if preferred_airline else 'Any'}. Class: {travel_class}.
        Then, use websurfer to browse the site where the url is created above, and list down the price for each option. 
        Ensure you scroll through the entire page by scrolling down too. Tell me which option is the best based on the list.
        """
        st.session_state.prompt = customized_prompt
        st.write(f"**Task is submitted with {selected_model} model.**")
        results = asyncio.run(collect_results(st.session_state.prompt, USE_AOAI, selected_model))
        st.session_state.elapsed = results[-1][1] if results[-1] is not None else None
    else:
        st.session_state.prompt = prompt

    # if st.button('Execute'):
    #     st.write(f"**Task is submitted with {selected_model} model.**")
    #     results = asyncio.run(collect_results(prompt, USE_AOAI, selected_model))
    #     st.session_state.elapsed = results[-1][1] if results[-1] is not None else None

    if st.session_state.elapsed is not None:
        st.write(f"**Prompt tokens: {st.session_state.prompt_token}**")
        st.write(f"**Completion tokens: {st.session_state.completion_token}**")

if __name__ == "__main__":
    main()