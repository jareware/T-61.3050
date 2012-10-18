import Image
def gen_static_data():
    c = 0
    for char in data.split("\n"):
        gen_thumbnail(char, c, ".")
        c += 1

def gen_all():
    c = 0
    for char in open("../../data/train.normalized.txt"):
        gen_thumbnail(char, c, "all")
        c += 1

def gen_thumbnail(char, c, path):
    img = Image.new("RGB", (8,16), 'white')
    line = char.split(",")
    for b in range(0,8):
     for a in range(0,16):
       if line[b+a*8+1] == "1":
        img.putpixel((b, a), (0,0,0))
    img.save("%s/img_%s_%s.png" % (path, chr(int(line[0])+97), c), "PNG")



f_labels = open("../test.solved.txt")
f_vectors = open("../test.normalized.txt")

c = 0
for line in f_vectors:
    c += 1
    label = f_labels.readline().strip()
    line = line.replace("127", label)
    gen_thumbnail(line, c, "all")
