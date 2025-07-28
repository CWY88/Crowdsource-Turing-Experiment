# """Query OpenAI Language Models.
# Functions for using OpenAI's API to query language models.
# Typical usage example:
#     ```
#     client = verify_openai_access(...)
#     model_settings = OpenAI_Model_Settings(...)
#     res = call_openai_api(
#         "Q: How many legs does a cat have",
#         model_settings,
#         client
#         )
#     ```
# """
# import openai
# from file_IO_handler import get_plaintext_file_contents
# import pathlib

# ENGINES = [
#     # "gpt-3.5-turbo-instruct",
#     "babbage-002"
#     # "davinci-002"
#     # "gpt-4",
#     # "gpt-3.5-turbo",
#     # "gpt-4-0613",  # All cheap, modern options
# ]

# def verify_openai_access(
#     path_to_organization: pathlib.Path,
#     path_to_api_key: pathlib.Path,
# ) -> openai.OpenAI:
#     """Return an OpenAI client using credentials from local files."""
#     organization = get_plaintext_file_contents(path_to_organization)
#     api_key = get_plaintext_file_contents(path_to_api_key)
#     client = openai.OpenAI(api_key=api_key, organization=organization)
#     return client

# class OpenAIModelSettings:
#     """Instance to track language model parameters before querying."""
#     def __init__(
#         self,
#         engine: str,
#         max_tokens: int = 0,
#         temperature: float = 1,
#         n: int = 1,
#         logprobs: int = 2,
#         echo: bool = True,
#         presence_penalty: float = 0,
#         frequency_penalty: float = 0,
#         stop=None,
#         params_descriptor: str = "no-complete-logprobs",
#     ) -> None:
#         if engine not in ENGINES:
#             raise ValueError(
#                 f"engine {engine} is not a valid choice of OpenAI engines {*ENGINES,}"
#             )
#         self.engine = engine
#         self.max_tokens = max_tokens
#         self.temperature = temperature
#         self.n = n
#         self.logprobs = logprobs
#         self.echo = echo
#         self.presence_penalty = presence_penalty
#         self.frequency_penalty = frequency_penalty
#         self.stop = stop
#         self.params_descriptor = params_descriptor
        
#     def __str__(self) -> str:
#         return str(vars(self))

# def call_openai_api(
#     prompt: str,
#     model_settings: OpenAIModelSettings,
#     client: openai.OpenAI,
# ) -> dict:
#     """Wrapper for getting OpenAI response using an explicit client."""
#     output = client.completions.create(
#         model=model_settings.engine,
#         prompt=prompt,
#         max_tokens=model_settings.max_tokens,
#         temperature=model_settings.temperature,
#         n=model_settings.n,
#         logprobs=model_settings.logprobs,
#         echo=model_settings.echo,
#         presence_penalty=model_settings.presence_penalty,
#         frequency_penalty=model_settings.frequency_penalty,
#         stop=model_settings.stop,
#     )
    
#     # Convert the OpenAI response object to a JSON-serializable dictionary
#     output_dict = output.model_dump()  # or output.dict() for older versions
    
#     return {"model": vars(model_settings), "output": output_dict}

"""Query OpenAI Language Models using Chat Completions API.
Functions for using OpenAI's Chat Completions API to query language models.
Typical usage example:
    ```
    client = verify_openai_access(...)
    model_settings = OpenAIModelSettings(...)
    res = call_openai_chat_api(
        "Q: How many legs does a cat have",
        model_settings,
        client
        )
    ```
"""
import openai
from file_IO_handler import get_plaintext_file_contents
import pathlib

# Chat completion models (removed legacy completion models)
MODELS = [
    "gpt-4",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "gpt-4o",
    "gpt-4o-mini"
]

def verify_openai_access(
    path_to_organization: pathlib.Path,
    path_to_api_key: pathlib.Path,
) -> openai.OpenAI:
    """Return an OpenAI client using credentials from local files."""
    organization = get_plaintext_file_contents(path_to_organization)
    api_key = get_plaintext_file_contents(path_to_api_key)
    client = openai.OpenAI(api_key=api_key, organization=organization)
    return client

class OpenAIModelSettings:
    """Instance to track language model parameters before querying."""
    def __init__(
        self,
        model: str,  # Changed from 'engine' to 'model'
        max_tokens: int = 1000,
        temperature: float = 0.3,
        n: int = 1,
        presence_penalty: float = 0.1,
        frequency_penalty: float = 0.1,
        stop=None,
        params_descriptor: str = "autism-community-response",
    ) -> None:
        if model not in MODELS:
            raise ValueError(
                f"model {model} is not a valid choice of OpenAI models {*MODELS,}"
            )
        self.model = model  # Changed from engine to model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.n = n
        # Removed logprobs and echo as they're not supported in chat completions
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.stop = stop
        self.params_descriptor = params_descriptor
        
    def __str__(self) -> str:
        return str(vars(self))

def call_openai_chat_api(
    prompt: str,
    model_settings: OpenAIModelSettings,
    client: openai.OpenAI,
) -> dict:
    """Wrapper for getting OpenAI chat completion response using an explicit client."""
    
    # Convert prompt to chat messages format
    messages = [{"role": "user", "content": prompt}]
    
    # Create chat completion parameters
    chat_params = {
        "model": model_settings.model,
        "messages": messages,
        "max_tokens": model_settings.max_tokens,
        "temperature": model_settings.temperature,
        "n": model_settings.n,
        "presence_penalty": model_settings.presence_penalty,
        "frequency_penalty": model_settings.frequency_penalty,
    }
    
    # Add stop parameter if specified
    if model_settings.stop is not None:
        chat_params["stop"] = model_settings.stop
    
    # Call the chat completions API
    output = client.chat.completions.create(**chat_params)
    
    # Convert the OpenAI response object to a JSON-serializable dictionary
    output_dict = output.model_dump()  # or output.dict() for older versions
    
    return {"model": vars(model_settings), "output": output_dict}

# Backward compatibility function that maps to chat API
def call_openai_api(
    prompt: str,
    model_settings: OpenAIModelSettings,
    client: openai.OpenAI,
) -> dict:
    """Backward compatibility wrapper that calls the chat completions API."""
    return call_openai_chat_api(prompt, model_settings, client)