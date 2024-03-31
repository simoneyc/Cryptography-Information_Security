def encryption(text, key):
    input_text = ""
    for char in text:
        if char.isalpha():
            shift = key % 26
            if char.islower():
                input_text += chr((ord(char) - ord('a') + shift) % 26 + ord('a'))
            elif char.isupper():
                input_text += chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
        else:
            input_text += char
    return input_text.upper()

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
    return ciphertext.lower()


text = input("input: ")
while True:
    try:
        key = int(input("input your key(int): "))
        break
    except ValueError:
        print("The key must be an integer!!!")

encrypted_text = encryption(text, key)
print("encrypt:", encrypted_text)

decrypted_text = decryption(encrypted_text, key)
print("decrypt:", decrypted_text)


