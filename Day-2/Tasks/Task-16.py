#While Loop

for i in range(0, 10, 2):
    print(i)

# print('Infinite while Loop')
# age = 28
#
# # the test condition is always True
# while age > 19:
#     print('Infinite Loop')

print('while loop with continue statement')
# Prints all letters except 'e' and 's'
i = 0
a = 'geeksforgeeks'

while i < len(a):
    if a[i] == 'e' or a[i] == 's':
        i += 1
        continue

    print(a[i])
    i += 1

print('while loop with break statement')
i = 0
a = 'geeksforgeeks'
while i < len(a):
    if a[i] == 'e' or a[i] == 's':
        i += 1
        break

    print(a[i])
    i += 1


print('while loop with pass statement')
a = 'geeksforgeeks'
i = 0
while i < len(a):
    i += 1
    pass

print('Value of i :', i)


print('while loop with else')
i = 0
while i < 4:
    i += 1
    print(i)
else:  # Executed because no break in for
    print("No Break\n")

i = 0
while i < 4:
    i += 1
    print(i)
    break
else:  # Not executed as there is a break
    print("No Break")
