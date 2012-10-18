import json
f = open("test.txt")
fout = open("test.normalized.txt", "w")



for line in f:
    line = line.split()
    char = chr(int(line[0]) + 97)
    data = line[1:]
    data = [int(i) for i in data]

    cd = []
    for b in range(0,16):
        d = []
        for c in range(0,8):
            d.append(data[b*8+c])
        cd.append(d)

    """
    for a in range(0,16):
       char_found = False
       for item in cd[0]:
           if item != 0:
               char_found = True
       if char_found:
           break
           # There's something on first line. Done.
       first = cd.pop(0)
       cd.append(first)
    """

    for a in range(0,16):
        char_found = False
        for b in range(0,16):
            if cd[b][0] != 0:
                char_found = True
        if char_found:
            break
        # Shift left
        new_cd = []
        for row in cd:
            first = row.pop(0)
            row.append(first)
            new_cd.append(row)
        cd = new_cd
#    cd = zip(*cd)

    fout.write(line[0])
    for row in cd:
       fout.write(","+",".join(map(str, row)))
    fout.write("\n")

fout.close()
