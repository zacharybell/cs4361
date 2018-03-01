from sklearn.cluster import KMeans
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics.cluster import entropy

people = fetch_lfw_people(min_faces_per_person=10, resize=0.7)

print(entropy(people.data))
