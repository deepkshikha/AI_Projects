import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from wordcloud import WordCloud
import re
import string
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

def load_and_preprocess_data(filepath):
    # Load data
    df = pd.read_csv(filepath, encoding='latin1')
    df.rename(columns={'v1':'class','v2':'sms'}, inplace=True)
    df = df[["class","sms"]]
    return df

def plot_class_distribution(df):
    sns.countplot(x='class', data=df, palette="viridis")
    plot.title("Class Distribution (Spam vs Non-Spam)")
    plot.xlabel("class")
    plot.ylabel("Count")
    plot.show()

def plot_word_count_distribution(df):
    df['word_count'] = df["sms"].apply(lambda x: len(x.split()))
    sns.histplot(data=df, x='word_count', hue='class', bins=30, kde=True, palette="husl")
    plot.title("class vs word_count")
    plot.xlabel("class")
    plot.ylabel("word_count")
    plot.show()

def generate_spam_wordcloud(df):
    spam_text = df[df['class'] == 'spam']['sms'].str.cat(sep=' ')
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(spam_text)
    plot.figure(figsize=(10, 6))
    plot.imshow(wordcloud, interpolation='bilinear')
    plot.axis('off')
    plot.title("Common Words in Spam Emails")
    plot.show()

def remove_tag(text):
    pattern = re.compile('<[^>]+>')
    return pattern.sub(r'', text)

def remove_urls(text):
    pattern = re.compile(r'\b(?:https?|ftp|www)\S+\b')
    return pattern.sub(r'', text)

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # Transport & map symbols
        u"\U0001F700-\U0001F77F"  # Alchemical symbols
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Chess Symbols
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U00002702-\U000027B0"  # Dingbats
        u"\U000024C2-\U0001F251"  # Enclosed characters
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def get_chat_words_dict():
    return {
        
        "AFAIK": "As Far As I Know",
        "AFK": "Away From Keyboard",
        "ASAP": "As Soon As Possible",
        "ATK": "At The Keyboard",
        "ATM": "At The Moment",
        "A3": "Anytime, Anywhere, Anyplace",
        "BAK": "Back At Keyboard",
        "BBL": "Be Back Later",
        "BBS": "Be Back Soon",
        "BFN": "Bye For Now",
        "B4N": "Bye For Now",
        "BRB": "Be Right Back",
        "BRT": "Be Right There",
        "BTW": "By The Way",
        "B4": "Before",
        "B4N": "Bye For Now",
        "CU": "See You",
        "CUL8R": "See You Later",
        "CYA": "See You",
        "FAQ": "Frequently Asked Questions",
        "FC": "Fingers Crossed",
        "FWIW": "For What It's Worth",
        "FYI": "For Your Information",
        "GAL": "Get A Life",
        "GG": "Good Game",
        "GN": "Good Night",
        "GMTA": "Great Minds Think Alike",
        "GR8": "Great!",
        "G9": "Genius",
        "IC": "I See",
        "ICQ": "I Seek you (also a chat program)",
        "ILU": "ILU: I Love You",
        "IMHO": "In My Honest/Humble Opinion",
        "IMO": "In My Opinion",
        "IOW": "In Other Words",
        "IRL": "In Real Life",
        "KISS": "Keep It Simple, Stupid",
        "LDR": "Long Distance Relationship",
        "LMAO": "Laugh My A.. Off",
        "LOL": "Laughing Out Loud",
        "LTNS": "Long Time No See",
        "L8R": "Later",
        "MTE": "My Thoughts Exactly",
        "M8": "Mate",
        "NRN": "No Reply Necessary",
        "OIC": "Oh I See",
        "PITA": "Pain In The A..",
        "PRT": "Party",
        "PRW": "Parents Are Watching",
        "QPSA?": "Que Pasa?",
        "ROFL": "Rolling On The Floor Laughing",
        "ROFLOL": "Rolling On The Floor Laughing Out Loud",
        "ROTFLMAO": "Rolling On The Floor Laughing My A.. Off",
        "SK8": "Skate",
        "STATS": "Your sex and age",
        "ASL": "Age, Sex, Location",
        "THX": "Thank You",
        "TTFN": "Ta-Ta For Now!",
        "TTYL": "Talk To You Later",
        "U": "You",
        "U2": "You Too",
        "U4E": "Yours For Ever",
        "WB": "Welcome Back",
        "WTF": "What The F...",
        "WTG": "Way To Go!",
        "WUF": "Where Are You From?",
        "W8": "Wait...",
        "7K": "Sick:-D Laugher",
        "TFW": "That feeling when",
        "MFW": "My face when",
        "MRW": "My reaction when",
        "IFYP": "I feel your pain",
        "TNTL": "Trying not to laugh",
        "JK": "Just kidding",
        "IDC": "I don't care",
        "ILY": "I love you",
        "IMU": "I miss you",
        "ADIH": "Another day in hell",
        "ZZZ": "Sleeping, bored, tired",
        "WYWH": "Wish you were here",
        "TIME": "Tears in my eyes",
        "BAE": "Before anyone else",
        "FIMH": "Forever in my heart",
        "BSAAW": "Big smile and a wink",
        "BWL": "Bursting with laughter",
        "BFF": "Best friends forever",
        "CSL": "Can't stop laughing"
        }

def chat_conversion(text):
    chat_words = get_chat_words_dict()
    new_text = []
    for word in text.split():
        if word.upper() in chat_words:
            new_text.append(chat_words[word.upper()])
        else:
            new_text.append(word)
    return " ".join(new_text)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    new_text = [word for word in text.split() if word not in stop_words]
    return ' '.join(new_text)

def remove_punc(text):
    exclude = string.punctuation
    for char in exclude:
        text = text.replace(char, '')
    return text

class LogisticRegression:
    def __init__(self, learning_rate=0.006, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.losses = []
    
    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def compute_loss(self, y_true, y_pred):
        epsilon = 1e-9
        y1 = y_true*np.log(y_pred+epsilon)
        y2 = (1-y_true)*np.log(1-y_pred+epsilon)
        return -np.mean(y1+y2)
    
    def feed_forward(self, X):
        z = np.dot(X, self.weights)+self.bias
        A = self._sigmoid(z)
        return A
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iters):
            A = self.feed_forward(X)
            cur_loss = self.compute_loss(y, A)
            self.losses.append(cur_loss)
            
            if i % 1000 == 0:
                print(f"Iteration {i}, Loss: {cur_loss}")
                
            dz = A - y
            dw = (1/n_samples) * np.dot(X.T, dz)
            db = (1/n_samples) * np.sum(dz)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
        plt.plot(range(self.n_iters), self.losses, label='Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss vs. Iteration')
        plt.legend()
        plt.show()
    
    def predict(self, X):
        threshold = 0.5
        y_hat = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(y_hat)
        y_predicted_cls = [1 if i > threshold else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

def main():
    # Load and preprocess data
    df = load_and_preprocess_data("../spam.csv")
    
    # Visualizations
    plot_class_distribution(df)
    plot_word_count_distribution(df)
    generate_spam_wordcloud(df)
    
    # Text preprocessing
    nltk.download('stopwords')
    df['sms'] = df['sms'].str.lower()
    df['sms'] = df['sms'].apply(remove_tag)
    df['sms'] = df['sms'].apply(remove_urls)
    df['sms'] = df['sms'].apply(remove_emojis)
    df['sms'] = df['sms'].apply(chat_conversion)
    df['sms'] = df['sms'].apply(remove_stopwords)
    df['sms'] = df['sms'].apply(remove_punc)
    
    # Prepare data for modeling
    label_encoder = LabelEncoder()
    df['class'] = label_encoder.fit_transform(df['class'])
    
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['sms']).toarray()
    y = df['class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate model
    LogReg = LogisticRegression()
    LogReg.fit(X_train, y_train)
    y_pred = LogReg.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_pred, y_test)}")

if __name__ == "__main__":
    main()