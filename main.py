import os
from dotenv import load_dotenv
import nest_asyncio
import torch
import ast
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel
from code_reader import code_reader
from prompts import context, code_parser_template

# Load environment variables and apply nest_asyncio
load_dotenv()
nest_asyncio.apply()

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set up LlamaParse
LlamaParse_api_key = os.getenv("LLAMA_CLOUD_API_KEY")
parser = LlamaParse(api_key=LlamaParse_api_key, result_type="markdown")

# Set up document reader
file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(input_files=["data/readme.pdf"], file_extractor=file_extractor).load_data()

# Set up Ollama embedding and LLM
embed_model = OllamaEmbedding(
    model_name="llama3.1:8b",
    base_url="http://localhost:11434",
    ollama_additional_kwargs={"mirostat": 0},
)

llm = Ollama(model="llama3.1:8b", request_timeout=300.0)

# Configure global settings
Settings.llm = llm
Settings.embed_model = embed_model

# Create index and query engine
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Set up tools
tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="api_documentation",
            description="this gives documentation about code for an API. use this for reading docs for the api."
        )
    ),
    code_reader,
]

# Set up agent
code_llm = Ollama(model="codellama", request_timeout=300.0)
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)

# Define output model
class CodeOutput(BaseModel):
    code: str
    description: str
    file_name: str

# Set up output parser and query pipeline
parser = PydanticOutputParser(CodeOutput)
json_prompt_str = parser.format(code_parser_template)
json_prompt_tmpl = PromptTemplate(json_prompt_str)
output_pipeline = QueryPipeline(chain=[json_prompt_tmpl, llm])

# Main loop
while (prompt := input("Enter a prompt (q to quit): ")).lower() != "q":
    retries = 0
    while retries < 3:
        try:
            result = agent.query(prompt)
            next_result = output_pipeline.run(response=result)
            
            # Parse the output correctly
            cleaned_json = parser.parse(next_result)
            break
        except Exception as e:
            retries += 1
            print(f"Error occurred, retry #{retries}:", e)
    
    if retries == 3:
        print("Unable to process request, try again.")
        continue
    
    print("Code generated:")
    print(cleaned_json.code)
    print("\nDescription:", cleaned_json.description)
    
    file_name = cleaned_json.file_name
    
    try:
        os.makedirs("output", exist_ok=True)
        with open(os.path.join("output", file_name), "w") as f:
            f.write(cleaned_json.code)
        print(f"Saved file: {file_name}")
    except Exception as e:
        print(f"Error saving file: {e}")