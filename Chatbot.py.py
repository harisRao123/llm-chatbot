from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace,HuggingFacePipeline
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
load_dotenv()
chat_history=[
   SystemMessage(content="you are helpful assstant")
]
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    device=-1,
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)
model=ChatHuggingFace(llm=llm)
while True:
    user_input=input("You:")
    if user_input.lower()=="exit":
     print("chat ended .")
     break;
    chat_history.append(HumanMessage(content=user_input))
    result=model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI:",result.content)