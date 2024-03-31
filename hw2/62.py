def encryption(text, key):
    input_text = ""
    for item in text:
        if item.isalpha():
            shift = key % 26
            if item.islower():
                input_text += chr((ord(item) - ord('a') + shift) % 26 + ord('a'))
            elif item.isupper():
                input_text += chr((ord(item) - ord('A') + shift) % 26 + ord('A'))
        else:
            input_text += item
    return input_text
                
def decryption(text, key):
    ciphertext = ""
    for char in text:
        if char.isalpha():
            shift = key % 26
            if char.islower():
                ciphertext += chr((ord(char) - ord('a') - shift) % 26 + ord('a'))
            elif char.isupper():
                ciphertext += chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
        else:
            ciphertext += char
    return 
# main
text = input("input: ")
try:
    key = int(input("input your key(int): "))
except ValueError:
    print("The key must be int!!!")

encrypted_text = encryption(text, key)
print("encrypt:", encrypted_text)

decrypted_text = decryption(encrypted_text, key)
print("decrypt:", decrypted_text)