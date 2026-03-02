#For Loop.

for i in range(0, 10, 2):
    print(i)


li = ["eat", "sleep", "repeat"]

for i, j in enumerate(li):
    print (i, j)


for i in range(1, 4):
    for j in range(1, 4):
        print(i, j)