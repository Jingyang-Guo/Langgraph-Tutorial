import json
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_core.globals import set_debug

# set_debug(True)

model = init_chat_model(
    model="qwen3.5-9b",
    model_provider="openai",
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

json_schema = {
    "title": "Movie",
    "description": "A movie with details",
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "The title of the movie"
        },
        "year": {
            "type": "integer",
            "description": "The year the movie was released"
        },
        "director": {
            "type": "string",
            "description": "The director of the movie"
        },
        "rating": {
            "type": "number",
            "description": "The movie's rating out of 10"
        }
    },
    "required": ["title", "year", "director", "rating"]
}

model_with_structure = model.with_structured_output(
    json_schema,
    method="json_schema",
    include_raw=True
)

response = model_with_structure.invoke("请你详细介绍电影夏洛特烦恼")