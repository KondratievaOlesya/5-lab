import os
from PIL import Image
path = r"H:\Рабочий стол\Python\biometry5.1\db"

for file_name in os.listdir(path):
    img = Image.open("db\\" + file_name)

    resized_img = img.resize((64*2, 48*2), Image.ANTIALIAS)
    # Имя файла и его формат
    base_name, ext = os.path.splitext(file_name)
    resized_img.save(base_name + ext)
