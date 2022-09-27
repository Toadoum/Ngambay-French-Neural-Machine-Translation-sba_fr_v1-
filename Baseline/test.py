text_to_translate = "les enfants partent au fleuve pour nager"
model_inputs = tokenizer(text_to_translate, return_tensors="pt")

# translate to French
from main import model
gen_tokens = model.generate(**model_inputs, forced_bos_token_id=tokenizer.get_lang_id("wo"))
print(tokenizer.batch_decode(gen_tokens, skip_special_tokens=True))