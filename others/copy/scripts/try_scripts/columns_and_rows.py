import numpy as np

data = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen"]

words_to_id = {words: id for id, words in enumerate(data)}
vectors = {coordinates: x for x in range(4)  for y in range(4) for z in range(4) for coordinates in [(x, y, z)]}
graph = {words: coordinates for words, coordinates in zip(words_to_id, vectors)}

coor = np.reshape(graph, (1, 1))
print(coor)

#am I in a right way for creating a dimensional array that stores a words with their unique id and location in 3d?
#what I want is to embed the words with the unique array with the graph location.