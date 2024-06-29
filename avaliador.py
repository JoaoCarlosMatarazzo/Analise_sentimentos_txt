import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Exemplos em português
data = {
    'review': [
        'Esse produto é ótimo e funciona perfeitamente.',
        'O filme foi muito ruim e chato.',
        'Adorei a comida, estava deliciosa.',
        'O atendimento foi péssimo, nunca mais volto.',
        'A entrega foi rápida e eficiente.',
        'Não gostei do serviço, deixou a desejar.',
        'Muito bom, recomendo a todos!',
        'Horrível, não recomendo para ninguém.'
    ],
    'sentiment': ['positivo', 'negativo', 'positivo', 'negativo', 'positivo', 'negativo', 'positivo', 'negativo']
}
data = pd.DataFrame(data)

def preprocess_text(text):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import string
    
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('portuguese'))
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

data['review'] = data['review'].apply(preprocess_text)
X_train, X_test, y_train, y_test = train_test_split(data['review'], data['sentiment'], test_size=0.3, random_state=42)
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(X_train_counts, y_train)
y_pred = model.predict(X_test_counts)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)
