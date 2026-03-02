import math

def displacement(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # in kilometers



def pyhtagoras(lat1, lon1, lat2, lon2):
    R = 6371

    x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    y = math.radians(lat2 - lat1)

    return R * math.sqrt(x*x + y*y)


# Example
print(displacement(31.464042679657418, 74.44210418319426, 31.454092808708573, 74.29139241188636))  # LGU → Johar Town C Block
print(pyhtagoras(31.464042679657418, 74.44210418319426, 31.454092808708573, 74.29139241188636))



def speed(distance, time):
    return distance / time

speed_ = speed(22.8,0.5)

print('Speed will be: {0}'.format(speed_))

# def displacement_time(displacement, speed):
#     return speed/displacement


# displace_ = pyhtagoras(31.464042679657418, 74.44210418319426, 31.454092808708573, 74.29139241188636)

# time = displacement_time()
print('The time needed to cover the displacement is : ')