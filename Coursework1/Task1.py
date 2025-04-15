from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import hashlib
import string
import random


def ds_hash(message: str) -> int:
    hash_value = 0
    for ch in message:
        hash_value = (hash_value * 71) + ord(ch)

    return hash_value & 0x7FFFFFFF
    
# To test if the hash function ds_hash() is strong collision resistant I used a technique called the "birthday attack".
def myAttack() -> bool:
    # The "birthday attack" I implemented works by creating a random word which is made up from a alphabet of random
    # numbers and lowercase and uppercase letters with the size of 64 bytes. Then hash the value using ds_hash() and check if the hash_value
    # is in the hashes that have already been seen if it is then return false however if it not then add the hash_value to the already seen hashes. 
    # It does this for this for num_trials time where num_trials is the length of the alphabet (62) to the power of the length of the message.
    lengthMessage = 64
    # This creates a set to store the already seen hashes so we can check if a new hash value is already been seen
    seen_hashes = set()
    # creates an alphabet of the lowercase and uppercase letters
    alphabet = string.ascii_lowercase + string.ascii_uppercase
    # This for loop adds the numbers from 0 to 9 to alphabet
    for i in range(10):
        alphabet = alphabet + str(i)
    num_trials = len(alphabet) ** lengthMessage 
    for _ in range(num_trials):
        # creates a random message of length lengthMessage using characters from alphabet to hash to check if the hash has already been seen
        message = ''.join(random.choices(alphabet, k=lengthMessage))
        hash_value = ds_hash(message)
        if hash_value in seen_hashes:
            return False  # Collision found
        # if there was no collision add the hash value to seen_hashes which store all the hashes we have already checked
        seen_hashes.add(hash_value)

    return True  # No collision found within the given number of trials

if __name__ == "__main__":
    print( myAttack() )