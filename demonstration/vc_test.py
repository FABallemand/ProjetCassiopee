from time import sleep
from subprocess import call

for i in range(0, 101, 10):
    print(i)
    call(["amixer", "-D", "pulse", "sset", "Master", str(i)+"%"])
    sleep(1)

for i in range(100, 0, -10):
    print(i)
    call(["amixer", "-D", "pulse", "sset", "Master", str(i)+"%"])
    sleep(1)

# from time import sleep
# import pyvolume

# for i in range(0, 101, 10):
#     print(i)
#     pyvolume.custom(percent=i)
#     sleep(1)

# for i in range(100, 0, -10):
#     print(i)
#     pyvolume.custom(percent=i)
#     sleep(1)