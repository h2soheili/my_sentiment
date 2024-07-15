#./llama_cpp/lib/llama-server -m ./bv/model.gguf --port 8080
import llama_cpp
llm = llama_cpp.Llama(
    model_path='./bv/model.gguf',
    n_ctx=16000,  # Context length to use
    n_threads=32,  # Number of CPU threads to use
    n_gpu_layers=0  # Number of model layers to offload to GPU
)

generation_kwargs = {
    "max_tokens": 20000,
    "stop": ["</s>"],
    "echo": False,  # Echo the prompt in the output
    "top_k": 1,  # This is essentially greedy decoding, since the model will always
    # return the highest-probability token. Set this value > 1 for sampling decoding
}

prompt = "The meaning of life is "
res = llm(prompt, **generation_kwargs)  # Res is a dictionary

print(res)

# from transformers import AutoTokenizer, AutoModelForCausalLM
#
# model_id = "./bv"
# filename = "model.gguf"
# prompt = "The meaning of life is "
#
# tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
# model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename)
#
# print(model(prompt))

