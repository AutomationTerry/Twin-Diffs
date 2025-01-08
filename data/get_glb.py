import os
import requests

filename_list_path = r"E:\Terry\code\twin-diffusions\data\prompt_train.txt"
save_directory = r"E:\Terry\code\twin-diffusions\data\objaverse_5k_prompt"

os.makedirs(save_directory, exist_ok=True)
downloaded_count = 0
url_prefix = "https://hf-mirror.com/datasets/allenai/objaverse/resolve/main/"

url_list = []

with open(
    r"E:\Terry\code\twin-diffusions\data\object-pathstry.txt",
    "r",
) as file:
    for line in file:
        line = line.strip()
        url_list.append(url_prefix + line + "?download=true")


for url in url_list:
    filename = os.path.basename(url)
    filename = os.path.basename(filename.split("?")[0])
    if filename in os.listdir(save_directory):
        print(f"文件 {filename} 已存在，跳过下载。")
        url_list.remove(url)
        continue

    response = requests.get(url)

    if response.status_code == 200:
        save_path = os.path.join(save_directory, filename)

        with open(save_path, "wb") as save_file:
            save_file.write(response.content)

        downloaded_count += 1
        print(f"已下载 {downloaded_count} 个文件：{filename}")

print(f"总共下载了 {downloaded_count} 个文件。")
