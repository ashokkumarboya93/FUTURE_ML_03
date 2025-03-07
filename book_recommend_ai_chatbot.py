import nltk
import warnings
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
# Suppress warnings
warnings.filterwarnings("ignore")

# Download the 'punkt_tab' data
nltk.download('punkt_tab', quiet=True)  # Download punkt_tab for sentence tokenization

# Suppress NLTK download logs
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load dataset
df = pd.read_csv("/content/drive/MyDrive/ashok/books.csv", sep=";", encoding="ISO-8859-1", on_bad_lines="skip", low_memory=False)

# Basic preprocessing
df = df[['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']].dropna()
df['combined_features'] = df['Book-Title'] + ' ' + df['Book-Author'] + ' ' + df['Publisher']

# Precompute TF-IDF for the entire dataset
vectorizer = TfidfVectorizer(stop_words='english')
book_vectors = vectorizer.fit_transform(df['combined_features'])

# Initialize ChatBot
chatbot = ChatBot("BookBot")
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train("chatterbot.corpus.english")

def preprocess_text(text):
    """Preprocess user input using NLTK."""
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words('english')]
    return ' '.join(filtered_tokens)

def recommend_books(user_input):
    """Recommends books based on user input."""
    genre_keywords = {
        "mystery": ["mystery", "detective", "crime", "thriller"],
        "romance": ["love", "romance", "relationship", "passion"],
        "adventure": ["adventure", "explore", "action", "journey"],
        "comedy": ["comedy", "funny", "humor", "satire"],
        "fantasy": ["fantasy", "magic", "dragons", "mythology"],
        "horror": ["horror", "scary", "ghost", "haunted"],
        "fiction": ["fiction", "novel", "story", "literature"],
        "history": ["history", "historical", "biography", "past"],
        "science fiction": ["sci-fi", "space", "aliens", "technology","science"],
        "self-help": ["self-help", "motivation", "success", "life lessons"],
        "psychology": ["psychology", "mind", "behavior", "mental health"],
        "philosophy": ["philosophy", "wisdom", "thinking", "existence"],
        "business": ["business", "entrepreneur", "startup", "finance"],
        "poetry": ["poetry", "poems", "verses", "literary"],
        "children": ["children", "kids", "young readers", "bedtime stories"]
    }

    user_input = preprocess_text(user_input)
    matching_genres = [genre for genre, keywords in genre_keywords.items() if any(word in user_input for word in keywords)]

    if matching_genres:
        filtered_df = df[df['Book-Title'].str.contains('|'.join(matching_genres), case=False, na=False)]
    else:
        return "I couldn't find anything related to your input. Please try again with a book title, author, or genre!"

    if filtered_df.empty:
        return "I couldn't find books related to that. Try mentioning a book title, author, or genre!"

    book_vectors_filtered = vectorizer.transform(filtered_df['combined_features'])
    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, book_vectors_filtered)

    book_indices = similarity.argsort()[0][-5:][::-1]
    recommendations = filtered_df.iloc[book_indices][['Book-Title', 'Book-Author', 'Year-Of-Publication']]
    return recommendations

def chatbot_interaction():
    print("\nWelcome to BookBot! Tell me your favorite genre, author, or book title, and I'll recommend some books!")

    greetings = ["hi", "hello", "hey", "good morning", "good evening", "namaste"]

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("BookBot: Happy reading! See you next time! 📚")
            break

        if any(word in user_input.lower() for word in greetings):
            print("BookBot: Hello! How can I assist you today? 😊")
            continue

        book_keywords = ["book", "novel", "author", "read", "recommend"]
        if any(word in user_input.lower() for word in book_keywords):
            recommendations = recommend_books(user_input)
            if isinstance(recommendations, str):
                print(f"BookBot: {recommendations}")
            else:
                print("\nBookBot: Here are some recommendations for you:")
                for index, row in recommendations.iterrows():
                    print(f"- {row['Book-Title']} by {row['Book-Author']} (Year: {row['Year-Of-Publication']})")
            continue

        print("BookBot: I couldn't find anything related to your input. Please try again with a book title, author, or genre!")

if __name__ == "__main__":
    chatbot_interaction()