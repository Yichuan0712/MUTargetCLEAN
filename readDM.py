import json

def read_data_from_file(filename):
    with open(filename, 'r') as file:
        data = file.read()
        dictionary = json.loads(data)
    return dictionary

# 假设你的文件名为"data.txt"
filename = "distance_map.txt"
data_dict = read_data_from_file(filename)
print(data_dict)
