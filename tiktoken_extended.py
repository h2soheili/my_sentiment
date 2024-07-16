import tiktoken

from tiktoken_trained import fn3, ENDOFTEXT_TOKEN, VOCAB_SIZE, ENDOFTEXT

trained_enc = fn3()
# print("VOCAB_SIZE" , VOCAB_SIZE)
# print(trained_enc.mergeable_ranks)
# print(trained_enc.pat_str)

enc = tiktoken.encoding_for_model("gpt2")
enc._pat_str = trained_enc.pat_str
enc._mergeable_ranks = trained_enc.mergeable_ranks
enc.max_token_value = VOCAB_SIZE

# enc = tiktoken.encoding_for_model("gpt2")
# enc = tiktoken.encoding_for_model("gpt-4o")


# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
extended_encoding = tiktoken.Encoding(
    name="bv_v0",
    pat_str=enc._pat_str,
    # explicit_n_vocab=enc.max_token_value + 1,
    mergeable_ranks={
        **enc._mergeable_ranks,
    },
    special_tokens={
        ENDOFTEXT: ENDOFTEXT_TOKEN,
    }
)

print("extended_encoding.max_token_value", extended_encoding.max_token_value)
