import sys
import codecs
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from sklearn.feature_extraction.text import TfidfVectorizer

# Παράδειγμα κειμένων
corpus = [
    'Αυτό είναι ένα παράδειγμα κειμένου.',
    'Κείμενο παράδειγμα δεύτερο.',
    'Τρίτο παράδειγμα για την δοκιμή.',
    'Αυτό είναι ένα παράδειγμα παράδειγμα.'
]

# Αρχικοποίηση του TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Υπολογισμός των TF-IDF των κειμένων
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# Εκτύπωση του σχήματος του πίνακα TF-IDF
print("Σχήμα TF-IDF πίνακα:")
print(tfidf_matrix.shape)

# Εκτύπωση των όρων που χρησιμοποιούνται ως features
print("\nΛεξιλόγιο (features):")
print(tfidf_vectorizer.get_feature_names_out())

# Εκτύπωση του πίνακα TF-IDF
print("\nTF-IDF Πίνακας:")
print(tfidf_matrix.toarray())


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Execute PCA to reduce the dimensionality of the TF-IDF matrix to 3 dimensions
pca = PCA(n_components=3)
tfidf_matrix_pca = pca.fit_transform(tfidf_matrix.toarray())

# Plot the TF-IDF matrix in 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(tfidf_matrix_pca[:, 0], tfidf_matrix_pca[:, 1], tfidf_matrix_pca[:, 2], alpha=0.5)

# Set labels and title
ax.set_title('TF-IDF Matrix Plot in 3D')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

# Show plot
plt.show()
