datasets for training:

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("allenai/tulu-3-sft-mixture")




from datasets import load_dataset

ds = load_dataset("nvidia/OpenMathInstruct-2")


from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("inclusionAI/Ling-Coder-SFT")


from datasets import load_dataset

ds = load_dataset("allenai/tulu-3-sft-personas-instruction-following")

from datasets import load_dataset

ds = load_dataset("allenai/tulu-3-sft-personas-instruction-following")

from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("allenai/WildChat-4.8M")




from datasets import load_dataset

ds = load_dataset("HumanLLMs/Human-Like-DPO-Dataset")









# for pandas

import pandas as pd

df = pd.read_parquet("hf://datasets/allenai/tulu-3-sft-personas-instruction-following/data/train-00000-of-00001.parquet")






import pandas as pd

df = pd.read_json("hf://datasets/HumanLLMs/Human-Like-DPO-Dataset/data.json")