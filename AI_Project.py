#Libraries
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#Exploration Data
def explore_dataset(data):
    #calculating distrubution
    sentiment_counts = data['sentiment'].value_counts()
    total_posts = data.shape[0]

    # Calculating percentages
    sentiment_percentages = (sentiment_counts / total_posts) * 100

    # Displaying
    print("\nSentiment Distribution:")
    for sentiment, count, percentage in zip(sentiment_counts.index, sentiment_counts, sentiment_percentages):
        print(f"{sentiment}: {count} posts ({percentage:.2f}%)")

#Extracting features
def extract_features(data):
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(data['text'])
    return features

# Splitting Dataset into training and testing
def split_dataset(features, labels):
    train_data, temp_data, train_labels, temp_labels = train_test_split(features, labels, test_size=0.3, random_state=42)
    val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5, random_state=42)
    return train_data, val_data, test_data, train_labels, val_labels, test_labels

#Training and Testing Data
def train_and_test(train_data, val_data, test_data, train_labels, val_labels, test_labels):
    model = LogisticRegression()
    model.fit(train_data, train_labels)
    
    # Predict labels for validation set
    val_predictions = model.predict(val_data)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    print(f"Validation Accuracy: {val_accuracy:.2f}")
    # Predict labels for test set
    test_predictions = model.predict(test_data)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print(f"Test Accuracy: {test_accuracy:.2f}")


# Data Preprocessing
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize the text into words
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize the words
    stopwords_set = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords_set]
    
    # Join the preprocessed tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text


#Reading dataset
df = pd.read_csv('DatasetAI.csv') 
#Analyzing sentiment distribution
sentiment_counts = df['sentiment'].value_counts()
print("Sentiment Distribution:")
print(sentiment_counts)
print()
#Visualizing sentiment distribution
plt.figure(figsize=(6, 4))
sentiment_counts.plot(kind='bar')
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()
#displaying train and test results
print("Training and Validation Accuracy:")
features = extract_features(df)
train_data, val_data, test_data, train_labels, val_labels, test_labels = split_dataset(features, df['sentiment'])
train_and_test(train_data, val_data, test_data, train_labels, val_labels, test_labels)
#Analyzing frequent words
all_text = ' '.join(df['text'])
wordcloud = WordCloud(width=800, height=400, max_words=50, background_color='white').generate(all_text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Most Frequent Words")
plt.show()

#Exploring hashtags and mentions and displaying them
hashtags = []
mentions = []
for text in df['text']:
    hashtags.extend([tag for tag in text.split() if tag.startswith('#')])
    mentions.extend([mention for mention in text.split() if mention.startswith('@')])

hashtag_counts = pd.Series(hashtags).value_counts().head(10)
mention_counts = pd.Series(mentions).value_counts().head(10)

print("Top 10 Hashtags:")
print(hashtag_counts)
print("\nTop 10 Mentions:")
print(mention_counts)
