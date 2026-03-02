# Using enumerate()

words = ['The', 'Big', 'Bang', 'Theory']

for index, value in enumerate(words):
    print(index, value)


#unzip
print('Unzip')
names = ['Leo', 'Kendall', 'Harry']
ages = (24, 27, 25)

for name, age in zip(names, ages):
    print(f"Name: {name}, Age: {age}")

print('Using items')

p1 = {'Washington': 'First', 'Lincoln': 'Emancipator', 'Roosevelt': 'Trust-Buster'}

for name, title in p1.items():
    print(name, title)

#using sorted

num = [1, 3, 5, 6, 2, 1, 3]

print("Sorted list:")
for n in sorted(num):
    print(n, end=" ")

print("\nSorted list without duplicates:")
for n in sorted(set(num)):
    print(n, end=" ")

#using reversed
num = [1, 3, 5, 6, 2, 1, 3]

for n in reversed(num):
    print(n, end=" ")

#While loop with if
count = 0

while count < 5:
    if count == 3:
        print("Count is 3")
    count += 1