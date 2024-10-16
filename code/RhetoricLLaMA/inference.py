from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("../../data/RhetoricLLaMA/test.csv")

def preprocess_case(text):
    max_tokens = 1000
    tokens = text.split(' ')
    num_tokens_to_extract = min(max_tokens, len(tokens))
    text1 = ' '.join(tokens[-num_tokens_to_extract:len(tokens)])
    return text1

for i,row in tqdm(df.iterrows()):
    input = row['Text']
    input = preprocess_case(input)
    df.at[i,'Text'] = input

model_id = "../../saved_models/RhetoricLLaMA/"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", token= ".....")
tokenizer = AutoTokenizer.from_pretrained(model_id)

def create_prompt(text):
    prompt = f""" ### Instructions:
    Analyze the given legal sentence and predict its rhetorical role as a number: None-0, Facts-1, Issue-2, Arguments of Petitioner-3, Arguments of Respondent-4, Reasoning-5, Decision-6.
    Note: The response must only contain a number between 0 and 6 representing the label of sentence. \
  
  ### Input:
  case_proceeding: <{case_pro}>

  ### Response:
  """

    return prompt

df["llama_p"] = ""
for i, row in tqdm(df.iterrows(), total=len(df)):
    case_pro = row["Text"]
    prompt = create_prompt(case_pro)
    input_ids = tokenizer(prompt, return_tensors='pt',truncation=True).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, max_new_tokens=100,)
    output = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
    df.at[i,"llama_p"] = output
df.to_csv("pred_41-50.csv", index = False)
