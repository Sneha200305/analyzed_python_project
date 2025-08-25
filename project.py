import pandas as pd
from textblob import TextBlob
import matplotlib
matplotlib.use('Agg')  # Prevent Tkinter errors in PyCharm
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Step 1: Load CSV file
df = pd.read_csv('customer_reviews.csv')  # Make sure this file is in the same folder
print("Data Preview:")
print(df.head())

# Step 2: Drop missing reviews
df.dropna(subset=['Review'], inplace=True)

# Step 3: Define sentiment analyzer
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Step 4: Apply sentiment function
df['Sentiment'] = df['Review'].apply(get_sentiment)

# Step 5: Show results
print("\nReviews with Sentiment:")
print(df[['Review', 'Sentiment']])

# Step 6: Save result to new CSV
df.to_csv('analyzed_reviews.csv', index=False)
print("✅ Analyzed data saved as 'analyzed_reviews.csv'")

# Step 7: Bar chart
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Sentiment', hue='Sentiment', palette='pastel', legend=False)
plt.title('Sentiment Distribution - Bar Chart')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.tight_layout()
plt.savefig('sentiment_chart.png')
print("✅ Sentiment bar chart saved as 'sentiment_chart.png'")

# Step 8: Word Clouds
positive_reviews = ' '.join(df[df['Sentiment'] == 'Positive']['Review'])
negative_reviews = ' '.join(df[df['Sentiment'] == 'Negative']['Review'])

wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)
wordcloud_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_reviews)

# Positive Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off')
plt.title("Positive Reviews Word Cloud")
plt.tight_layout()
plt.savefig('positive_wordcloud.png')
print("✅ Positive word cloud saved as 'positive_wordcloud.png'")

# Negative Word Cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
plt.title("Negative Reviews Word Cloud")
plt.tight_layout()
plt.savefig('negative_wordcloud.png')
print("✅ Negative word cloud saved as 'negative_wordcloud.png'")

# Step 9: Pie Chart
sentiment_counts = df['Sentiment'].value_counts()
labels = sentiment_counts.index
sizes = sentiment_counts.values

plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#8bc34a', '#f44336', '#ffc107'])
plt.title('Sentiment Distribution - Pie Chart')
plt.axis('equal')
plt.tight_layout()
plt.savefig('sentiment_piechart.png')
print("✅ Sentiment pie chart saved as 'sentiment_piechart.png'")


# test change
