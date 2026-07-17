def calculator(expression):
    try:
        return eval(expression)
    except:
        return "Invalid expression"

def keyword_extractor(text):
    words = text.split()
    return list(set(words))

def general_response(query):
    return f"General Response: {query}"
