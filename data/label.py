import os


def get_result(text):
    import replicate
    os.environ['REPLICATE_API_TOKEN'] = "r8_eNKLF27FBXIOkjvrkNMWtU2aLM7L5YI0OlLBB"
    text = text.replace('\n', "")
    text = text.strip()
    # The meta/meta-llama-3-70b-instruct model can stream output as it's running.
    rep_stream = replicate.stream(
        "meta/meta-llama-3-70b-instruct",
        input={
            "top_k": 50,
            "top_p": 0.9,
            "prompt": text,
            "max_tokens": 512,
            "min_tokens": 0,
            "temperature": 0.6,
            "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "presence_penalty": 1.15,
            "frequency_penalty": 0.2
        },
    )
    for event in rep_stream:
        print(str(event), end="")


t = """

برای سهامدار بازار بورس این خبر به کدام حالت از حالت های  { خرید قوی - خرید - خنثی - فروش - فروش قوی } اشاره دارد ؟
فقط حالت را بنویس . توضیح ننویس .

نرخ سود همانطور که مشاهده می شود با کاهش 0.66 % در نرخ بهره بازار بین بانکی شاهد کاهش 0.95 % در نرخ قرارداد ریپو و کاهش 2.5 % در میانگین نرخ بازده اوراق با درآمد ثابت هستیم. بررسی منحنی بازده استخراج شده از نماد های اخزا نشان می دهد که میانگین نرخ های بازده در کلیه سررسیدها کاهشی بودند و به 22.59 % رسیده اند.

"""

get_result(t)
