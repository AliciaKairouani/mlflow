from stackapi import StackAPI


site = StackAPI('stackoverflow')


questions = site.fetch('questions',  
    min=5,  # Score > 5
    views=10,  # ViewCount > 10
    answers=1,  # AnswerCount > 0
    filter= 'withbody'
    
)

print(questions)
