from anthropic import AnthropicVertex
from os import getenv
from litellm import completion

from requests import post, Response
from google.auth import default
from google.auth.transport.requests import Request


model_id:str = getenv("MODEL_ID", "claude-sonnet-5")
project_id:str=getenv("VERTEX_PROJECT_ID", "gcp-project")
region:str=getenv("VERTEX_LOCATION", "global")


def anthropic() -> None:
    client:AnthropicVertex = AnthropicVertex(region=region, project_id=project_id)
    message = client.messages.create(
        model=model_id,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Hello Claude! How are you running on GCP?"}
        ]
    )
    print(message.content[0].text)


def litellm() -> None:
    # docs: https://docs.litellm.ai/docs/providers/vertex
    response = completion(
        model=f"vertex_ai/{model_id}",
        messages=[{"role": "user", "content": "Hello! How are you doing this great day!"}],
        vertex_project=project_id,
        vertex_location=region
    )
    print(response.choices[0].message.content)


def raw() -> None:
    # 1. Automatically fetch your active Google Cloud OAuth token
    # credentials, project_id = default()
    credentials, _ = default()
    credentials.refresh(Request())
    auth_token:str = credentials.token

    # check out model cards: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/partner-models/use-partner-models

    # 2. Structure the Vertex AI Claude endpoint URL (global endpoint)
    url:str = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/publishers/anthropic/models/{model_id}:rawPredict"

    # 3. Headers require your Bearer token and Anthropic's structural version
    headers:dict[str, str] = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json; charset=utf-8"
    }

    # 4. Standard Anthropic Messages payload
    payload:dict[str, object] = {
        "anthropic_version": "vertex-2023-10-16",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 1024
    }

    # 5. Execute the agnostic HTTP request
    response:Response = post(url, headers=headers, json=payload)
    # print(response.content)
    print(response.json()["content"][0]["text"])


if __name__ == "__main__":
    anthropic()
    litellm()
    raw()
