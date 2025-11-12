import pandas as pd

# 手动下载数据集放到 scripts/data，处理成jsonl文件
# https://huggingface.co/datasets/inclusionAI/SWE-CARE/blob/main/data/dev-00000-of-00001.parquet
# https://huggingface.co/datasets/inclusionAI/SWE-CARE/blob/main/data/test-00000-of-00001.parquet

# 转换test数据集
test_df = pd.read_parquet('data/test-00000-of-00001.parquet')
test_df.to_json('swe_care_test.jsonl', orient='records', lines=True)

# 转换dev数据集
dev_df = pd.read_parquet('data/dev-00000-of-00001.parquet')
dev_df.to_json('swe_care_dev.jsonl', orient='records', lines=True)
