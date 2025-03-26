import csv
import difflib
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# File path
file_path = r"C:\Users\Admin\AppData\Local\Programs\Python\Python312-32\Scripts\drugs1.csv"

# Load dataset
if not os.path.exists(file_path):
    print("Error: File not found. Check the file path and try again.")
    exit()

questions = []
answers = []

# Read CSV file
try:
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            questions.append(row[0].strip())  # Strip spaces
            answers.append(row[1].strip())
except Exception as e:
    print(f"Error: {e}")
    exit()

# Convert questions into numerical form using TF-IDF
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# Function to find the best answer
def get_answer(user_question):
    user_vector = vectorizer.transform([user_question])  # Convert input question to vector
    similarities = cosine_similarity(user_vector, question_vectors)  # Compute similarity
    best_match_index = similarities.argmax()  # Get index of the best match
    
    if similarities[0, best_match_index] > 0.3:  # Similarity threshold
        return answers[best_match_index]
    else:
        return "Sorry, I don't understand your question."

# Chatbot loop
while True:
    user_input = input("Ask me a question: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    response = get_answer(user_input)
    print("AI:", response)
