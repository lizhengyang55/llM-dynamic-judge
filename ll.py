# check_output_structure.py
import os
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="agentscope-ai/OpenJudge",
    filename="train_rm/grpo/pairwise/train.parquet",
    repo_type="dataset",
)

df = pd.read_parquet(path)

# 深入看 output 结构
for i in range(3):
    print(f"\n{'='*70}")
    print(f"Row {i}")
    print(f"{'='*70}")
    
    row = df.iloc[i]
    
    # input
    inp = row['input']
    print(f"\ninput (len={len(inp)}):")
    for j, turn in enumerate(inp):
        print(f"  turn[{j}]: role={turn['role']}, content={str(turn['content'])[:150]}...")
    
    # output
    out = row['output']
    print(f"\noutput (len={len(out)}):")
    for j, item in enumerate(out):
        print(f"\n  output[{j}]:")
        answer = item['answer']
        print(f"    answer.role: {answer.get('role', 'N/A')}")
        print(f"    answer.content: {str(answer['content'])[:200]}...")
        label = answer.get('label', {})
        print(f"    answer.label: {label}")
    
    print(f"\nposition_variant: {row['position_variant']}")
    print(f"unique_id: {row['unique_id']}")
    print(f"subset: {row['subset']}")
    print(f"task_category: {row['task_category']}")

# 统计 position_variant 分布
print(f"\n\n{'='*70}")
print("position_variant 分布:")
print(df['position_variant'].value_counts())
print(f"\nsubset 分布:")
print(df['subset'].value_counts())
print(f"\ntask_category 分布:")
print(df['task_category'].value_counts())
print(f"\nsource 分布:")
print(df['source'].value_counts())

# 看 output 中 label 的所有可能值
print(f"\n\n{'='*70}")
print("检查 label.preference 的所有值:")
preferences = set()
for i in range(len(df)):
    out = df.iloc[i]['output']
    for item in out:
        pref = item['answer'].get('label', {}).get('preference', 'N/A')
        preferences.add(pref)
print(f"  所有 preference 值: {preferences}")

# 看 label.response_id
print("\n检查 label.response_id 的值 (前5行):")
for i in range(5):
    out = df.iloc[i]['output']
    for j, item in enumerate(out):
        label = item['answer'].get('label', {})
        print(f"  row[{i}].output[{j}]: response_id={label.get('response_id')}, preference={label.get('preference')}, strength={label.get('preference_strength')}")