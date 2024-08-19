from transformers import PreTrainedTokenizerFast

from ai.tiktoken_trained import get_tokenizer

t1 = PreTrainedTokenizerFast(tokenizer_file='./tokenizer_trained.json',
                             padding_side='right',
                             truncation_side='right',
                             bos_token='<|begin_of_text|>',
                             eos_token='<|end_of_text|>',
                             pad_token='<|end_of_text|>',
                             )

print(t1.pad_token_id)

t2 = get_tokenizer(show_log=False)
import tiktoken

t3 = tiktoken.get_encoding("o200k_base")

texts = [
    "طیب نیا رانت ناشی از اختلاف قیمت ارز باعث فساد است",
    "مهم برای فملی قیمت پایه کاتد در بورس کالا تغییر کرد smbroker",
    "کارشناس بازار پول و ارز همانگونه که رئیس کل بانک مرکزی بیان کردند می توان گفت که ارز 28500 با 4200 با هم قابل مقایسه نیست. ارز 28500 تومانی را بانک مرکزی با نرخ 27500 تومان از دولت خریداری می کند و با 1000 تومان بالاتر به فروش می رساند. بنابراین تغییری در متغیر پایه پولی رخ نمی دهد. ارز 28500 تومانی صرفا ارزی است که دولت از طریق فروش نفت و گاز در اختیار دارد و به بانک مرکزی می فروشد. مابقی ارزها شامل ارز پتروشیمی ها...",
    "آلمان واردات نفت از ایران را تکذیب کرد اکانت رسمی وزارت خارجه آلمان, در توییتی اضمن تکذیب خبر مربوط به صادرات نفت خام ایران به این کشور نوشت بر اساس اطلاعات ما, این یک گزارش نادرست به یورو استات است. از سوی دیگر آژانس بین المللی انرژی قبلاً این خطا را تصحیح کرده است."

]

for txt in texts:
    print("text:")
    print(txt)
    print("hf tokenizer trained  vocab_size:", t1.vocab_size)

    res = t1.encode(txt)
    print(res)
    print([t1.decode(i) for i in res])
    print("token count", len(res))
    print(t1.decode(res))

    print("_______")
    print("tiktoken trained vocab_size:", t2.n_words)

    res2 = t2.encode(txt, bos=False, eos=False)
    print([t2.decode([i]) for i in res2])
    print(res2)
    print("token count ", len(res2))
    print(t2.decode(res2))

    print("_______")
    print("tiktoken gpt-4o vocab_size:", t3.n_vocab)

    res3 = t3.encode(txt)
    print([t3.decode([i]) for i in res3])
    print(res3)
    print("token count ", len(res3))
    print(t3.decode(res3))

    print("_____________________________")
