import os
import numpy
import pandas
import json
import PIL
import subprocess
import torch
import re
import pandas as pd
import numpy as np
from datasets import Dataset
from unsloth import FastLanguageModel
from vllm import SamplingParams
from trl import GRPOConfig, GRPOTrainer
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"


max_seq_length = 4096
lora_rank = 48

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B",
    max_seq_length = max_seq_length,
    load_in_4bit = False,
    fast_inference = True,
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.9,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ],
    lora_alpha = lora_rank*2,
    use_gradient_checkpointing = "unsloth",
    random_state=42,
)


THINK_START  = "<think>"
THINK_END    = "</think>"
ANSWER_START = "<answer>"
ANSWER_END   = "</answer>"

system_prompt = f"""Ты составляешь инвестиционный портфель на бирже на следующий год.

ФОРМАТ ПРОМПТА:
- В полученном тобой промпте будет содержаться информация о финансовых показателях компании за прошедший год.
- Для некоторых компаний будут содержаться выдержки из годового финансового отчета за прошедший год.
- Будет прогноз по данной акции: покупать/держать/продавать.
- Полученный тобой промпт УЖЕ заканчиватся токеном {THINK_START}. НЕ ДОБАВЛЯЙ {THINK_START}.

ОПРЕДЕЛЕНИЯ:
– Рентабельность собственного капитала - отношение чистой прибыли за период к средневзвешенной величине собственного капитала.
- EBITDA — аналитический показатель, равный объёму прибыли до вычета расходов по уплате налогов, процентов, и начисленной амортизации.
- CAPEX - это расходы компании на приобретение, создание или модернизацию долгосрочных физических активов (оборудование, недвижимость, транспорт, ПО).
- R&D - это расходы на исследования и разработки, направленные на создание новых продуктов, технологий или услуг, а также усовершенствование существующих. 

ФОРМАТ ОТВЕТА:
- Напиши короткое обоснование принятого тобой решения, а затем токен {THINK_END}.
- СРАЗУ ЖЕ ПОСЛЕ ЭТОГО напиши {ANSWER_START}ВЕРДИКТ{ANSWER_END}.
- ВЕРДИКТ - это ОДНО ИЗ ТРЕХ СЛОВ: покупать/держать/продавать.
- После {ANSWER_END} не пиши ничего.
"""

SP = json.dumps(system_prompt)

chat_template = (
    "{% if messages and messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] + '\\n' }}"
        "{% set loop_messages = messages[1:] %}"
    "{% else %}"
        "{{ " + SP + " + '\\n' }}"
        "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
        "{% if message['role'] == 'user' %}"
            "{{ message['content'] + '\\n' }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + eos_token + '\\n' }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
        "{{ '" + THINK_START + "\\n' }}"
    "{% endif %}"
)

tokenizer.chat_template = chat_template

# два EOS для Qwen
EOS_IDS = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
EOS_IDS = [x for x in EOS_IDS if isinstance(x, int) and x >= 0]
VERDICT_RE = r"покупать|держать|продавать"
ANS_TAG_RE = re.compile(r"<answer>\s*(" + VERDICT_RE + r")\s*</answer>", flags=re.DOTALL)

def to_float(x: str):
    try: return float(x)
    except: return None

def same_number(a: str, b: str, tol: float = 1e-9) -> bool:
    fa, fb = to_float(a), to_float(b)
    return (fa is not None) and (fb is not None) and (abs(fa - fb) <= tol)


ds = pd.read_csv('datasets/vencel/enriched-dataset-csv/enriched_dataset.csv', index_col = 0)
train = ds[ds["year"] <= 2023]
train["llm_target"] = train.groupby('year')['revenue'].rank(pct=True).apply(lambda x: "покупать" if x >= 0.8 else "держать" if x > 0.2 else "продавать")

def make_prompt(sample):
    text = (
        f"Тикер: {sample['ticker']}\n" +
        f"Доход от акции за предыдущий год: {(np.exp(sample['momentum'])* 100 - 100):.1f}%\n" +
        f"Рентабельность собственного капитала: {sample['ROE']:.1f}\n" +
        f"Cоотношение цены компании и ее прибыли: {sample['P/E']:.1f}\n" +
        f"Отношение цены компании к ее балансовой стоимости: {sample['P/BV']:.1f}\n" +
        f"Отношение стоимости компании к EBITDA: {sample['EV/EBITDA']:.1f}\n" +
        f"Отношение размера долга к EBITDA: {sample['Долг/EBITDA']:.1f}\n" +
        f"Отношение затрат на R&D к CAPEX: {sample['R&D/CAPEX']:.1f}\n" +
        f"Отношение CAPEX к выручке: {sample['CAPEX/Выручка']:.1f}\n" +
        f"\n"
        f"Прогноз: {sample['llm_target_predict']}"
        f"\n"
    )
    if len(sample["report"]) > 0:
        text += f"Выдержки из финансового отчета компании за год:\n"
        text += "\n\n".join([f"-{_}" for _ in sample["report"]])
    return text

train = Dataset.from_list([
    {"messages": [{"role": "user", "content": make_prompt(e)}], "gold": e.llm_target} 
        for e in train.iterrows()
])


def extract_pred(text: str) -> str:
    t = (text or "").replace(",", "")
    matches = ANS_TAG_RE.findall(t)
    if matches:
        return matches[-1].strip()
    nums = VERDICT_RE.findall(t)
    return nums[-1].strip() if nums else ""

def same_number(a: str, b: str, tol: float = 1e-9) -> bool:
    a = (a or "").strip()
    b = (b or "").strip()
    if a == "" or b == "":
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except:
        return False

def reward_correct_only(prompts, completions, answer, **kwargs):
    """
    Единственный reward:
    1.0 если гайденс на покупку/продажу был верный
    иначе 0.0
    """
    scores = []
    for c, gold in zip(completions, answer):
        pred = extract_pred(c[0]["content"])
        gold = str(gold).replace(",", "").strip()
        scores.append(1.0 if same_number(pred, gold) else 0.0)
    return scores


tokenizer.padding_side = "left"

vllm_sampling_params = SamplingParams(
    temperature=1.0,
    top_p=1.0,
    top_k=5,
    seed=42,
    stop=["<|im_end|>", tokenizer.eos_token],
    include_stop_str_in_output=True,
)

training_args = GRPOConfig(
    use_vllm=True,
    vllm_sampling_params=vllm_sampling_params,

    learning_rate=1e-5,
    weight_decay=0.0,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=10,

    per_device_train_batch_size=12,
    gradient_accumulation_steps=1,

    num_generations=5,
    max_prompt_length=3072,
    max_completion_length=1024,

    max_steps=500,
    save_steps=500,
    report_to="none",
    output_dir="outputs_grpo",
)


trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[reward_correct_only],
    args=training_args,
    train_dataset=train,
)
trainer.train()

model.save_pretrained("/ckpts")
tokenizer.save_pretrained("/ckpts")
