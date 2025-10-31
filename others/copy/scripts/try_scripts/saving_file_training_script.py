import re
import glob
import os

extension = ".txt"
folder_path = ".../.../.../something"  # Replace with actual path
search_pattern = os.path.join(folder_path, f"*{extension}")
reader = glob.glob(search_pattern)
temp_holder = []

for path in reader:
    with open(path, encoding="utf-8") as file:
        content = file.read()
        cleaner = re.findall(r"\w+\s\w+\b", content.lower())
        temp_holder.extend(cleaner)

# Prepare save location
save_loc = r".../.../.../save_here"  # Replace with actual path
os.makedirs(save_loc, exist_ok=True)

# Generate unique filename
increment = 0
while True:
    file_name = f"new_file_{increment}.txt"
    save_path = os.path.join(save_loc, file_name)
    if not os.path.exists(save_path):
        break
    increment += 1

# Save the content
with open(save_path, "w", encoding="utf-8") as output_file:
    output_file.write("\n".join(temp_holder))