path_to_text = './shakespeare.txt'
TRAIN_FRAC = 0.90
VAL_FRAC = 0.03
TEST_FRAC = 0.07

with open(path_to_text, 'r') as f:
    text = f.read()


# print some basic stats
print(f"number of characters: {len(text)}")

# convert all characters to uppercase
text = text.upper()
print(f"number of unique characters: {len(set(text))}")


assert TRAIN_FRAC + VAL_FRAC + TEST_FRAC == 1.0, "split fractions do not add up to 1!"


total_length = len(text)
num_train = int(TRAIN_FRAC * total_length)
num_val = int(VAL_FRAC * total_length)
num_test = int(TEST_FRAC * total_length)


train_set = text[:num_train]
val_set = text[num_train: num_train + num_val]
test_set = text[num_train + num_val: num_train + num_val + num_test]

print("\n")
print(f"Training Set Length: {len(train_set)}")
print(f"Validation Set Length: {len(val_set)}")
print(f"Test set Length: {len(test_set)}")

with open('./train.txt', 'w') as f:
    f.write(train_set)

with open('./val.txt', 'w') as f:
    f.write(val_set)

with open('./test.txt', 'w') as f:
    f.write(test_set)

