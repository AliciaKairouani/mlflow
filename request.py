from stackapi import StackAPI


site = StackAPI('stackoverflow')


questions = site.fetch('questions',  
    min=5,  # Score > 5
    views=10,  # ViewCount > 10
    answers=1,  # AnswerCount > 0
    filter= 'withbody'
    
)

print(questions)


# Fetch questions from Stack Overflow
questions = SITE.fetch('questions', tagged='python', sort='votes', order='desc')

filtered_questions = []

# Filter questions
for question in questions['items']:
    view_count = question.get('view_count', 0)
    score = question.get('score', 0)
    answer_count = question.get('answer_count', 0)
    tags = question.get('tags', [])

    if view_count > 10 and score > 5 and answer_count > 0 and len([tag for tag in tags if '<' in tag]) >= 5:
        filtered_questions.append(question)