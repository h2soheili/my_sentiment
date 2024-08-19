from tokenizers import Tokenizer, Regex

from tokenizers.trainers import BpeTrainer
from tokenizers import decoders, processors, normalizers, models, pre_tokenizers

corpus_files = ["../data/text.txt"]

tokenizer = Tokenizer(models.BPE())

special_tokens = [
    "<|begin_of_text|>",  # 0
    "<|end_of_text|>",  # 1
    # "<|padding_token|>",
    # "<space>",
]

tokenizer.eos_token = 1
tokenizer.pad_token = tokenizer.eos_token

tokenizer.normalizer = normalizers.Sequence([
    normalizers.Replace("``", '"'),
    normalizers.Replace("''", '"'),
    normalizers.Strip(),
    normalizers.NFD(),
    normalizers.Lowercase(),
    normalizers.StripAccents(),
    normalizers.Replace(Regex(" {2,}"), " "),
    # Replace(" ", "<space>")
])

# pat_str = "|".join(
#     [
#         r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
#         r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
#         r"""\p{N}{1,3}""",
#         r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
#         r"""\s*[\r\n]+""",
#         r"""\s+(?!\S)""",
#         r"""\s+""",
#     ]
# )

pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [pre_tokenizers.Digits(individual_digits=True),
     pre_tokenizers.Split(pat_str, "isolated"),
     pre_tokenizers.ByteLevel(add_prefix_space=False)])

# tokenizer.pre_tokenizer = PreTokenizerSequence([Digits(), pre_tokenizers.Whitespace()])
# tokenizer.pre_tokenizer = PreTokenizerSequence([Digits(individual_digits=True), Split(pat_str, "merged_with_previous")])

# tokenizer.pre_tokenizer = PreTokenizerSequence([Digits(individual_digits=True), Split(pat_str, behavior="removed")])

tokenizer.post_processor = processors.Sequence([processors.ByteLevel(trim_offsets=False), ])

tokenizer.decoder = decoders.ByteLevel()

trainer = BpeTrainer(
    # special_tokens=special_tokens,
    vocab_size=25_000,
    min_frequency=10,
    show_progress=True,
    max_token_length=20,

)

print("started...")
tokenizer.train(files=corpus_files, trainer=trainer)

tokenizer.add_special_tokens(special_tokens)

tokenizer.eos_token = "<|end_of_text|>"
tokenizer.pad_token = "<|end_of_text|>"
tokenizer.save("./tokenizer_trained.json")
print("...end")
# print(len(tokenizer))
