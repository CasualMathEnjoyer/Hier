file = "data/smallvoc_fr.txt"
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


