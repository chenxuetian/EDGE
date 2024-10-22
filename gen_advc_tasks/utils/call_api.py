import base64
# import openai
import anthropic
from dotenv import load_dotenv


load_dotenv(override=True)

# If you don't store your API key in an .env file, uncomment the following line and replace <API KEY> with your actual API key
# api_key = <API KEY>

# If your request need to be forwarded to a specific server, uncomment the following line and replace <BASE URL> with the URL
# base_url = <BASE URL>

client_anthropic = anthropic.Anthropic(
    # api_key=api_key,
    # base_url=base_url,
)
# client_openai = openai.OpenAI(
#     # api_key=api_key
#     # base_url=base_url,
# )


def base64_encode_image(img_path: str):
    """Given an image path, return the base64 encoding of the image."""
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def call_vision_api(
    model, system_prompt, user_prompt, img_path, *, temperature=0.7
) -> str:
    base64_image = base64_encode_image(img_path)

    # Anthropic format API
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": f"{base64_image}"
                    }
                },
                {
                    "type": "text",
                    "text": user_prompt
                }
            ]
        }
    ]
    response = client_anthropic.messages.create(
        model=model,
        messages=messages,
        max_tokens=2000,
        temperature=temperature,
    )
    return response.content[0].text

    # OpenAI format API
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": { "url": f"data:image/png;base64,{base64_image}" }
                }, 
                {
                    "type": "text",
                    "text": user_prompt
                }, 
            ]
        }
    ]
    response = client_openai.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
    )
    response_content = response.choices[0].message.content
    return response_content

    # token_usage = {"prompt": response.usage.prompt_tokens, "completion": response.usage.completion_tokens}
    # return response_content, token_usage

    return response.choices[0].message.content

# def print_total_cost(token_usages):
#     total_usage = {"prompt": 0, "completion": 0}
#     for token_usage in token_usages:
#         total_usage["prompt"] += token_usage["prompt"]
#         total_usage["completion"] += token_usage["completion"]
#     print("Total usage: ", total_usage, end="\t")
    
#     total_cost = total_usage["prompt"] / 1e6 * 3 +  total_usage["completion"] / 1e6 * 15
#     print(f"Total cost: ${total_cost}")
