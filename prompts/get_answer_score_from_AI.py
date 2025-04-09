def get_answer_score_from_AI_prompt(question, answer):
    return f"""
            You are an AI language model tasked with evaluating how relevant a given answer is to a specific question. Please carefully compare the QUESTION and ANSWER below. Use logical, semantic, and contextual understanding to decide how well the answer addresses the question.

            Your response must be a SINGLE INTEGER between 1 and 10, based on the following criteria:

            - 10 = Perfectly relevant and fully answers the question.
            - 8–9 = Mostly relevant, may miss very minor details.
            - 6–7 = Moderately relevant, addresses part of the question but lacks key context.
            - 4–5 = Minimally relevant, some surface-level connection but largely incomplete.
            - 2–3 = Barely relevant, vague or off-topic.
            - 1 = Completely irrelevant.

            DO NOT EXPLAIN. DO NOT ADD ANY TEXT. RETURN ONLY A SINGLE INTEGER (1–10) ON A NEW LINE.

            QUESTION: {question}

            ANSWER: {answer}
            """