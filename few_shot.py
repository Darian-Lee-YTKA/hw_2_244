from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "meta-llama/Llama-3.2-1B"


tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(model_id, use_auth_token=True)


pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


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





#processing the val data 
# =====================================================================================
# =====================================================================================
# =====================================================================================
#         .     .
#    ...  :``..':
#     : ````.'   :''::'
#   ..:..  :     .'' :
#``.    `:    .'     :
#    :    :   :        :
#     :   :   :         :
#     :    :   :        :
#      :    :   :..''''``::.
#       : ...:..'     .''
#       .'   .'  .::::'
#      :..'''``:::::::
#      '         `::::
#                  `::.
#                   `::
#                    :::.
#         ..:.:.::'`. ::'`.  . : : .
#       ..'      `:.: ::   :'       .:
#      .:        .:``:::  :       .: ::
#      .:    ..''     :::.'    ,.:   ::
#       : .''         .:: : :''      ::
#        :          .'`::
#                      ::
#                      ::
#                      ::
#                      ::
#                      ::
# little rose 
df = pd.read_json("val.json")

from ast import literal_eval
from sklearn.metrics import precision_score, recall_score, f1_score

all_true = []
all_pred = []

for i in range(len(df)):
    val_text = " ".join(df["text"][(i*10):((i+1)*10)])
    true_tags = df["tag"][(i*10):((i+1)*10)].tolist()
    true_starts = df["start"][(i*10):((i+1)*10)].tolist()
    true_ends = df["end"][(i*10):((i+1)*10)].tolist()

    full_prompt = prompt + f"\nText:\n{val_text}\n\nAnswer:\n"

    output = pipe(full_prompt, max_new_tokens=1024, do_sample=False)[0]["generated_text"]

    try:
        predicted_json_str = output[output.rfind("["):]
        predicted_spans = json.loads(predicted_json_str)
    except:
        print("❌ JSON parse error")
        continue


    gold = set(zip(true_starts, true_ends, true_tags))
    pred = set((d["start"], d["end"], d["tag"]) for d in predicted_spans)


    all_gold = list(gold)
    for span in all_gold:
        all_true.append(1)
        all_pred.append(1 if span in pred else 0)

    for span in pred:
        if span not in gold:
            all_true.append(0)
            all_pred.append(1)

precision = precision_score(all_true, all_pred)
recall = recall_score(all_true, all_pred)
f1 = f1_score(all_true, all_pred)

print(f"\nPrecision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1 Score:  {f1:.2%}")
