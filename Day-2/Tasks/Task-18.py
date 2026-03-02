for i in range(1, 11):
    if i == 6:
        continue
    print(i, end=" ")


#Nested Loops

a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

for row in a:
    for num in row:
        if num == 3:
            continue
        print(num, end=" ")


#While Loop

i = 0
while i < 10:
    if i == 5:
        i += 1  # ensure the loop variable is incremented to avoid infinite loop
        continue
    print(i)
    i += 1