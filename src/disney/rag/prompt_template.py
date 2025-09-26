"""Single prompt template for Disney customer experience questions."""

from langchain.prompts import ChatPromptTemplate


# Disney QA Template
DISNEY_QA_TEMPLATE = """You are an assistant for Disney customer experience questions.
Use the following pieces of retrieved context from Disney customer reviews to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}

Answer:"""


def get_prompt_template() -> ChatPromptTemplate:
    """Get the Disney QA prompt template.
    
    Returns:
        ChatPromptTemplate instance for Disney customer experience questions
    """
    return ChatPromptTemplate.from_template(DISNEY_QA_TEMPLATE)
