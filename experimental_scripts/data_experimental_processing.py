file = "../data/smallvoc_fr.txt"
fileF = "data/smallvoc_fr_.txt"

with open(file, "r", encoding="utf-8") as f:  # with spaces
    file = f.read()

with open(fileF, "w", encoding="utf-8") as f:  # with spaces
    for c in file:
        if c == " ":
            f.write("_ ")
        elif c == "\n":
            f.write("\n")
        else:
            f.write(c + " ")


train_in_file_name = "../data/smallvoc_fr_.txt"
train_out_file_name = "../data/smallvoc_en_.txt"
val_in_file_name = "../data/smallvoc_fr_.txt"
val_out_file_name = "../data/smallvoc_en_.txt"
test_in_file_name = "../data/smallervoc_fr_.txt"  # test input file
test_out_file_name = "../data/smallervoc_en_.txt"  # test target

with open(train_in_file_name, "r", encoding="utf-8") as f:  # with spaces
    source = f.read()
with open(train_out_file_name, "r", encoding="utf-8") as ff:
    target = ff.read()

sl = len(source.split("\n"))
tl = len(target.split("\n"))

print(sl)
print(sl/11)  # 10367
print(113800/10*8, 113800/10*2, 239)
print(113800/10*8 + 113800/10*2 + 239)

# source_list = source.split("\n")
# with open("fr_train.txt", "w", encoding="utf-8") as f:
#     with open("fr_val.txt", "w", encoding="utf-8") as v:
#         with open("fr_test.txt", "w", encoding="utf-8") as t:
#             for i in range (len(source_list)):
#                 if i < 113800/10*8:
#                     f.write(source_list[i])
#                     f.write("\n")
#                 elif i >= 113800/10*8 + 113800/10*2:
#                     t.write(source_list[i])
#                     t.write("\n")
#                 else:
#                     v.write(source_list[i])
#                     v.write("\n")