from stackapi import StackAPI

def request_api(score,views,filter):
    """permet de recuperer les données de l'Api stackoverflow"""
    site = StackAPI('stackoverflow')
    questions = site.fetch('questions',  
        min = score,  # Score > 5
        views = views,  # ViewCount > 10
        answers = 1,  # AnswerCount > 0
        filter = filter)
    return questions

test = request_api(score=5,views = 10, filter= 'withbody')
test