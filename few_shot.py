'''from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "meta-llama/Llama-3.2-1B"

# Загрузка токенизатора и модели
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=True)

# Создаём пайплайн генерации текста
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


prompt = """Your task is to perform BIO tagging to identify the following in math texts:
Here is a short description of each entity type:
• definition: text which define some new concept or object.
• theorem: text which make a rigorous, provable claim.
• proof: text which proves (or suggests a proof) to a theorem
• example: an example of a defined object or application of a theorem.
• name: the name of a newly defined object or a theorem (e.g., “abelian group”, “Theorem 1.2”)
• reference: a reference to a name which may have been defined previously.
Names and references (“identifiers”) can only appear within other entities.

-- Input
Here’s some stuff about mathy nonsense...
By \\(S \\subset \\mathbb{N}\\) having a /least element/, we mean that there exists
an \\(x \\in S\\), such that for every \\(y \\in S\\), we have \\(x \\leq y\\).
Here’s some more stuff...

-- Output
[O] [O] [O] ...
[definition] ... [definition, name] [definition, name] [definition] ...
[O] [O] [O] ...

-- Input
For every \\(\\varepsilon > 0\\), there exists \\(\\delta > 0\\) such that...
Then we can say that the function is continuous.

-- Output
[theorem] [theorem] [theorem] ... [theorem]
[O] [O] [O] [definition]


"""

# Добавим новый пример, чтобы модель сгенерировала разметку
prompt += """
-- Input
Let us now prove the above theorem.
Assume the contrary. Then there is...

-- Output
"""

# Генерация (можно настроить параметры генерации)
output = pipe(prompt, max_new_tokens=100, temperature=0.3, do_sample=False)[0]["generated_text"]

print(output)'''


import pandas as pd
import json
import random
df = pd.read_json("train.json")


# connecting them to make it match the testing data more 
texts = []
tags = []
starts = []
ends = []

for i in range(5):
    num = random.randint(1, 5) # to vary the few shot examples a lil bit (because I'm a cheeky little bitch *giggles*)
    j = (i+1) * 10 + num
    texts.append(" ".join(df["text"][(i*10):j]))
    tags.append(df["tag"][(i*10):j].tolist())
    starts.append(df["start"][(i*10):j].tolist())
    ends.append(df["end"][(i*10):j].tolist())



prompt = """You are an expert at extracting structured information from mathematical texts. Your task is to identify and label spans of text according to the following entity types:

Entity types:
• definition: A phrase or sentence that introduces and defines a new concept or object.
• theorem: A formal and provable statement (e.g., “Let G be a group…”).
• proof: A logical argument that demonstrates the truth of a theorem.
• example: An illustration of how a concept works or how a theorem can be applied.
• name: The specific name of a defined object or theorem (e.g., “Theorem 1.2”, “abelian group”).
• reference: A mention or citation of a previously defined object or theorem.
Note: Entities of type "name" and "reference" can only appear *within* other entities (e.g., a name inside a definition or theorem).

Your output should be a JSON array of dictionaries. Each dictionary should contain:
• "tag" — one of the entity types listed above,
• "start" — character offset (integer) where the span starts (inclusive),
• "end" — character offset (integer) where the span ends (exclusive).

Return only the list of tagged spans — do not return anything else.

Below are several examples:
"""

for i in range(5):
    prompt += f"\nText:\n{texts[i]}\n\nAnswer:\n"

    current_tags = tags[i] if isinstance(tags[i], list) else [tags[i]]
    current_starts = starts[i] if isinstance(starts[i], list) else [starts[i]]
    current_ends = ends[i] if isinstance(ends[i], list) else [ends[i]]

    spans = {
        "tags": current_tags,
        "starts": [int(s) for s in current_starts],
        "ends": [int(e) for e in current_ends]
    }

    prompt += json.dumps(spans, indent=2) + "\n"

prompt += "HERE IS YOUR INPUT: "



#processing the val data 
df = pd.read_json("val.json")
texts = []
tags = []
starts = []
ends = []

for i in range(5):
    num = random.randint(1, 5) # to vary the few shot examples a lil bit (because I'm a cheeky little bitch *giggles*)
    j = (i+1) * 10 + num
    texts.append(" ".join(df["text"][(i*10):j]))
    tags.append(df["tag"][(i*10):j].tolist())
    starts.append(df["start"][(i*10):j].tolist())
    ends.append(df["end"][(i*10):j].tolist())
