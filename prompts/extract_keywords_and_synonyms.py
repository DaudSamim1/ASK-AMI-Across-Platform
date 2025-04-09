def extract_keywords_and_synonyms_prompt(text):
    return f"""
    You are an AI-specialised interactive bot working for a major US law firm. Your task is to perform two steps on the provided text:

    1. Extract all keywords from the text which may be relevant to a legal case. Only include keywords that are present in the text.
    2. For each keyword, generate 2 synonyms or closely related terms that could help in legal query searches.

    Format your output exactly like this:
    Keywords: keyword1, keyword2, keyword3, ...
    Synonyms:
    keyword1: synonym1, synonym2
    keyword2: synonym1, synonym2
    keyword3: synonym1, synonym2

    Text:
    {text}
    """
