prompt_text = """
You are a knowledge‐graph extraction assistant. Your job is to extract meaningful relationships between pairs of entities and present them in a precise structured format.

**Instructions:**
- Read the input text carefully.
- Identify each key entity pairs and their relationships.
- For each relationship, output a single line using the following format (with no exceptions):

Entity1<|>Entity2<|>Relationship<|>Score

Where:
- `Entity1` and `Entity2` are two entities mentioned in the text.
- `Relationship` is the semantic connection between them (e.g., "works at", "is part of", "developed by").
- `Score` is an integer from 1 to 10 indicating confidence or relevance (10 = very strong).

**Formatting Rules:**
- Use the delimiter `<|>` exactly as shown.
- Separate each relationship record with `##`.
- Do not include bullet points, Markdown, JSON, or explanations.
- Only output plain lines as described.

**Example:**
Apple<|>Tim Cook<|>CEO<|>9##
Microsoft<|>Satya Nadella<|>CEO<|>9

Now extract all possible entities and relationships from the following text:

{input_text}

Remember:
- Only use the entity types from this list: {entity_types}
- Use <|COMPLETE|> at the very end of your response.
""".strip()

with open("ENTITY_EXTRACTION_PROMPT.txt", "w") as f:
    f.write(prompt_text)
