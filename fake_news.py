import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("news.csv")
if 'text' not in data.columns or 'label' not in data.columns:
    print("Error: Dataset must have 'text' and 'label'")
    exit()


def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

data['text'] = data['text'].apply(clean)


data['label'] = data['label'].astype(str).str.upper()


data = data[data['text'] != ""]


x = data['text']
y = data['label']


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


vectorizer = TfidfVectorizer(stop_words='english')
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)


model = SGDClassifier(
    loss='hinge',
    penalty=None,
    learning_rate='pa1',
    eta0=1.0,
    max_iter=1000
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


while True:
    news = input("\nEnter news (type exit to stop): ")
    
    if news.lower() == "exit":
        break
    
    news = clean(news)

    if news == "":
        print("Enter valid text!")
        continue

    vec = vectorizer.transform([news])
    pred = model.predict(vec)

    print("Prediction:", pred[0])
