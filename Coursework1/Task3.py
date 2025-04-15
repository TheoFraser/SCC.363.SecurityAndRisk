from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import hashlib
import string

def convertBytesToInt(value: bytes) -> int:
    # Convert the bytes object to its hexadecimal representation
    value_hex = value.hex()
    # Convert the hexadecimal string to an integer
    value_int = int(value_hex, 16)
    # Return the integer representation of the bytes object
    return value_int

def convertIntToBytes(value: int) -> bytes:
    # Convert the integer value to its hexadecimal representation
    value_hex = hex(value)[2:]
    # Pad with a leading zero if the length of the hex value is odd
    if len(value_hex) % 2 != 0:
        value_hex = '0' + value_hex  
    # Convert the hexadecimal string to bytes
    value_bytes = bytes.fromhex(value_hex)
    # Return the bytes representation of the integer value
    return value_bytes

def cioPKCS7(data, block_size): # block size in bits
    # Create a PKCS#7 padding object with the specified block size
    padder = padding.PKCS7(block_size).padder()
    # Update the padding object with the input data encoded as bytes
    padded_data = padder.update(data.encode())
    # Finalize the padding and add any remaining padding bytes
    padded_data += padder.finalize()
    return padded_data

def CustomAESMode(key: bytes, iv: bytes, plaintext: str) -> str:
    # To implement the AES mode in Figure 1 firstly we apply PKCS7 padding to the data ensuring that its length 
    # becomes a multiple of 128 bits. Then I split the data in blocks of 16 bytes (128 bits) as this AES mode requires the data
    # to be in blocks of 128 bits. Then I create a AES cipher with the mode ECB as each part of the ci.
    paddingText = cioPKCS7(plaintext, 128)
    splitted = [paddingText[i:i+16] for i  in range(0, len(paddingText), 16)]
    cipher = Cipher(algorithms.AES(key), modes.ECB())
    encryptor = cipher.encryptor()
    # Next I convert the iv value from bytes to a integer and convert the first 16 bytes (128 bits) of the plaintext to a Integer 
    # So I can XOR both of the them just like the diagram. Then I need to convert the XOR value into bytes as I need to update
    # the encryptor with the XOR value which will generate the cipherText for the first 16 bytes of the plaintext. Then I set the new 
    # value for iv_int to be the integer value for cipherText as in the diagram the new Initialisation vector becomes the cipherText. Then I add the 
    # cipherText to a variable called the totalCipherText which holds all the cipherTexts. Then i repeat this for all the 16 byte blocks of plainText.
    iv_int = convertBytesToInt(iv)
    totalCipherText = b''
    for word_bytes in splitted:
        word_int = convertBytesToInt(word_bytes)
        XORValue_int = iv_int ^ word_int
        XORValue_bytes = convertIntToBytes(XORValue_int)
        cipherText = encryptor.update(XORValue_bytes)
        iv_int = convertBytesToInt(cipherText)
        totalCipherText = totalCipherText + cipherText
    return totalCipherText.hex()

if __name__ == "__main__":
    key = bytes.fromhex("06a9214036b8a15b512e03d534120006") 
    iv = bytes.fromhex("3dafba429d9eb430b422da802c9fac41") 
    txt = "This is a text" 
    print( CustomAESMode(key, iv, txt) )   
