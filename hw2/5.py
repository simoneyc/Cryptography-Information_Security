def reverse_string(input_str):
    if not isinstance(input_str, str):
        raise TypeError("String!!!!")

    reversed_str = ""
    for i in range(len(input_str) - 1, -1, -1):
        reversed_str += input_str[i]
    return reversed_str


user_input = input("input: ")
try:
    reversed_input = reverse_string(user_input)
    print("result:", reversed_input)
except TypeError as e:
    print("錯誤:", e)
