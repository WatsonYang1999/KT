import os
import openai

openai.api_key = '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'

# List of code snippets you want to label
code_snippets = [
  "for i in range(5):\n    print(i)",
  "def add(a, b):\n    return a + b",
  "class Circle:\n    def __init__(self, radius):\n        self.radius = radius"
]

# Set of programming knowledge points or skills
knowledge_set = [
  "loops",
  "functions",
  "classes"
]

# Initialize a dictionary to store the responses
responses = {}

# Loop through each code snippet and call the OpenAI API
for idx, snippet in enumerate(code_snippets):
  prompt = f"Label the following code snippet for its programming knowledge point: '{snippet}'\nKnowledge points: {', '.join(knowledge_set)}\nLabel:"

  response = openai.Completion.create(
    engine="text-davinci-003",  # You can also use "text-davinci-003" for shorter completions
    prompt=prompt,
    max_tokens=1  # Adjust the number of tokens according to your needs
  )

  label = response.choices[0].text.strip()
  responses[snippet] = label

# Print the responses
for snippet, label in responses.items():
  print(f"Code Snippet: {snippet}\nLabel: {label}\n")
