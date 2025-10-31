import os
import glob
import re
import numpy as np
import pickle
import random

#preparing the paths for searching
folder_input_path = r"H:\OWN_AI\others\copy\train\inputs"

#this is what will gonna find, what extension should be find
extension = ".txt"
search_pattern = os.path.join(folder_input_path, f"*{extension}")
txt_files = glob.glob(search_pattern)
temp_storage = []

#get all the words from temp_storage and clean them then give a unique id to each word
for file_path in txt_files:
    with open(file_path, encoding="utf-8") as file:
        content = file.read()
    tokens = re.findall(r"\b\w+\b", content.lower())
    temp_storage.extend(tokens)

merged_words = temp_storage
cleaned_merged_words = sorted(set(merged_words))
words_to_id = {word: idx for idx, word in enumerate(cleaned_merged_words)}
tokens_id = [words_to_id[word] for word in merged_words]

#################################################################################

#for opening and reading the pickle file
file_path = r"H:\OWN_AI\others\copy\train\outputs\dictionary.pkl"
with open(file_path, "rb") as f:
    data = pickle.load(f)
    
# Word generator
word_generator_limit = 500
j = 0
arr = []

while True:
    me = input("Enter your word 'exit' to quit: ")

    random_reply = random.choice(list(data.keys()))
    arr.append(random_reply)
    j += 1

    if me == "exit":
        break
    else:
        print("Core:", arr)







