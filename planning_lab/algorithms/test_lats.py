from langchain_groq import ChatGroq
from planning_lab.algorithms.lats import lats
from planning_lab.algorithms.environment import Environment

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

environment = Environment(
    success_threshold=0.6
)

result = lats(
    task="Solve 2 + 2.",
    llm=llm,
    environment=environment,
    iterations=2,
    n_actions=2,
)

print("Success:", result.success)
print("Output:", result.output)
print("Best score:", result.best_score)
print("Iterations:", result.iterations)