print("ولشرق".encode())

import tiktoken

# enc = tiktoken.get_encoding("o200k_base")
# assert enc.decode(enc.encode("hello world")) == "hello world"

# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model("gpt-4o")  # o200k_base gpt-4o

print(enc.encode("ولشرق"))
# a = "ول شرق ولشرق در ماهه به سود 74 ریال ریا رسیده است".split(" ")
# for i in a:
#     print(i, " ", enc.encode(i))

# print(enc.encode_single_token("رسیده"), ">>>>")

i = enc.max_token_value
print("max_token_value", i)

new_items = [
    "ولشرق",
]
new_items_dict = dict()

for item in new_items:
    i += 1
    new_items_dict[item.encode(encoding="utf-8")] = i

extended_enc = tiktoken.Encoding(
    name="o200k_extended",
    pat_str=enc._pat_str,
    mergeable_ranks={
        **enc._mergeable_ranks,
        **new_items_dict,
    },
    special_tokens={
        **enc._special_tokens,
    }
)

print(extended_enc.encode("ولشرق"))
