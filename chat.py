import os
import openai
import os
openai.api_key = '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'


completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)

