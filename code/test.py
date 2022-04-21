result = []
res = []
file_list = ["train", "valid", "test"]
for filename in file_list:
    temp = []
    with open("../pretreatment/x_" + filename, 'r', encoding="utf-8") as f:
        for line in f:
            temp.append(len(line.split(' ')))
        result.append(max(temp))
        res.append(len([x for x in temp if x > 120]))

print(result)
print(res)
