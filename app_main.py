import streamlit as st

def get_related_tags(word):
    # Function to get related tags based on the input word (you can implement your logic here)
    # For example, using word embeddings, TF-IDF, or any other method to find related tags
    # Replace this with your logic to get related tags
    related_tags = ["Python", "Data Science", "Machine Learning", "Pandas"]
    return related_tags

# Title and Subtitle
st.title("Tags Related")
st.subheader("Your Word")

# Input box for the word
input_word = st.text_input("Enter your word")

# Button to trigger related tags calculation
if st.button("Get Related Tags"):
    if input_word:
        # Displaying related tags
        st.subheader("Tags Related")
        related_tags = get_related_tags(input_word)
        st.write(related_tags)
    else:
        st.warning("Please enter a word to get related tags.")