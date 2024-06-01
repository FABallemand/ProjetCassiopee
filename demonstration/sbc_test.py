from time import sleep
import screen_brightness_control as sbc

all_methods = sbc.get_methods()

for method_name, method_class in all_methods.items():
    print("================")
    print("Method:", method_name)
    print("Class:", method_class)
    print("Associated monitors:", sbc.list_monitors(method=method_name))
    print("================")

print(f"current brightness is : {sbc.get_brightness()}")

sleep(1)

new_brightness = 100
sbc.set_brightness(new_brightness)

sleep(1)

print(f"current brightness is : {sbc.get_brightness()}")

# while(True):
#     print(f"current brightness is : {get_brightness()}")
#     sleep(1)