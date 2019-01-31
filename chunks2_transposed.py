import pickle

chunks = pickle.load(open('chunks2.p','rb'))
new_chunks = []

for chunk in chunks:
    new_chunk = chunk[:-2]
    new_chunk.append('select-green')
    if chunk[3] and not chunk[5]:
        new_chunk.append(1)
    else:
        new_chunk.append(0)
    new_chunk.append('select-orange')
    if chunk[5] and not chunk[3]:
        new_chunk.append(1)
    elif chunk[3] and chunk[5]:
        new_chunk.append(1)
    else:
        new_chunk.append(0)
    new_chunk.append('select-around')
    new_chunk.append(0)

    new_chunks.append(new_chunk)

pickle.dump(new_chunks, open("chunks_transposed.p","wb"))
print("done")