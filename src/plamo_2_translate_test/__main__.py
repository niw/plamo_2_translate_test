import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline

model = AutoModelForCausalLM.from_pretrained("pfnet/plamo-2-translate", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("pfnet/plamo-2-translate", trust_remote_code=True)

device = "mps" if torch.mps.is_available() else "cpu"
pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=device)

while True:
    english = input("Type English Text: ")

    prompt = f"""<|plamo:op|>dataset
translation
<|plamo:op|>input lang=English
{english}
<|plamo:op|>output lang=Japanese
"""
    stop_string = "<|plamo:op|>"

    result = pipe(prompt, max_new_tokens=1024, stop_strings=stop_string, tokenizer=tokenizer)
    generated_text = result[0]["generated_text"]
    translation = generated_text[len(prompt):len(generated_text) - len(stop_string)]

    print(f"Translation: {translation}")
