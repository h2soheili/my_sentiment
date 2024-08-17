from transformers import PreTrainedTokenizerFast

from ai.tiktoken_trained import get_tokenizer

txt = "طیب نیا رانت ناشی از اختلاف قیمت ارز باعث فساد است"

print("_____________________________")
print("text:")
print(txt)
print("_____________________________")
print("hf tokenizer trained:")

t1 = PreTrainedTokenizerFast(tokenizer_file='./example_tokenizer.json')

res = t1.encode(txt)
print(res)
print([t1.decode(i) for i in res])
print("token count", len(res))

print("_____________________________")
print("tiktoken trained:")
t2 = get_tokenizer(show_log=False)
res2 = t2.encode(txt, bos=False, eos=False)
print([t2.decode([i]) for i in res2])
print(res2)
print("token count ", len(res2))

print("_____________________________")
print("tiktoken gpt-4o:")
import tiktoken

t3 = tiktoken.get_encoding("o200k_base")
res3 = t3.encode(txt)
print([t3.decode([i]) for i in res3])
print(res3)
print("token count ", len(res3))
