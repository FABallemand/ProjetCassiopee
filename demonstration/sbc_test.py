from time import sleep
import screen_brightness_control as sbc

all_methods = sbc.get_methods()

for method_name, method_class in all_methods.items():
    print("================")
    print("Method:", method_name)
    print("Class:", method_class)
    print("Associated monitors:", sbc.list_monitors(method=method_name))
    print("================")

brightness = sbc.get_brightness()
print(f"current brightness is : {brightness}")

for i in range(0, 101, 5):
    sbc.set_brightness(i)
    print(f"current brightness is : {sbc.get_brightness()}")
    sleep(0.5)

for i in range(100, -1, -5):
    sbc.set_brightness(i)
    print(f"current brightness is : {sbc.get_brightness()}")
    sleep(0.5)

sbc.set_brightness(brightness[0])