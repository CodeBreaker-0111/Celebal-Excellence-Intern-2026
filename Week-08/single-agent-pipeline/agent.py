from tools import calculator, keyword_extractor, general_response

def agent(query):
    q = query.lower()

    if q.startswith("calculate"):
        expr = query[len("calculate"):].strip()
        return calculator(expr)

    elif q.startswith("keywords"):
        text = query[len("keywords"):].strip()
        return keyword_extractor(text)

    return general_response(query)

if __name__ == "__main__":
    while True:
        query = input("Enter Query (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        print("Output:", agent(query))
