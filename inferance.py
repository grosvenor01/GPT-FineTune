import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, TextStreamer

prompt_format = """ Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

model = AutoPeftModelForCausalLM.from_pretrained(
    "lora_model",
    force_download=True
)

tokenizer = AutoTokenizer.from_pretrained("lora_model")

inputs = tokenizer(
[
    prompt_format.format(
        "if you're a doctore answer the question below : ", # instruction
        "i have some problem in my back what is probably the problem ?", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer #show token by token (instead of waiting the whole token to be generated )
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)