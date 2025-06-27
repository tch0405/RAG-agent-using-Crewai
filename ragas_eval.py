from ragas.metrics import answer_relevancy, context_precision, faithfulness
from ragas.evaluation import evaluate
from datasets import Dataset


with open("rag_output.txt", "r") as f:
    answer = f.read().strip()

dataset = Dataset.from_dict({
    "question": ["What is CrewAI and how does it help with AI workflows?"],
    "answer": [answer],
    "contexts": [[
        "CrewAI is a Python framework for orchestrating multiple AI agents...",
        "Each agent has a role, goal, and backstory to function autonomously."
    ]],
    "ground_truth": ["CrewAI helps manage LLM agents working collaboratively to solve complex problems."]
})

results = evaluate(
    dataset,
    metrics=[answer_relevancy, context_precision, faithfulness]
)

print("RAGAS Evaluation Results:")
print(results)
