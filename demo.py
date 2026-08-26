from openrouter import OpenRouter
from pydantic import BaseModel
import time
from datasets import load_dataset
import json

sample = load_dataset(
    "json",
    data_files="helpsteer2_sample_10.jsonl",
    split="train",
)


class StructuredOutput(BaseModel):
    correctness: int
    helpfulness: int
    explanation: str


prompt = """Evaluate the model response to the given user prompt.
Determine its correctness and helpfulness.
Assign an integer score from 0 to 4 for each.
Briefly explain your evaluation.
"""


def call_llm(user_prompt: str, model_response: str):
    with OpenRouter(api_key="key") as open_router:
        response = open_router.chat.send(
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}\n user prompt: {user_prompt.strip()} \n model_response: {model_response.strip()}",
                }
            ],
            model="google/gemini-3.7-flash",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "StructuredOutput",
                    "strict": True,
                    "schema": StructuredOutput.model_json_schema(),
                },
            },
            max_tokens=4096,
            provider={"zdr": True, "allow_fallbacks": False},
            reasoning={"effort": "low"},
            stream=False,
        )

    output_data = response.choices[0].message.content
    print("Structured Output:", output_data)
    return output_data


result_file = "score_first.json"
results = []

for row in sample:
    output = call_llm(row["prompt"], row["response"])

    results.append(
        {
            "user_prompt": row["prompt"],
            "model_response": row["response"],
            "evaluation": json.loads(output),
        }
    )

    time.sleep(3)

with open(result_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
