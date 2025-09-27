"""Prompt templates for Disney customer experience questions and query parsing."""

from langchain.prompts import ChatPromptTemplate


# Disney QA Template
DISNEY_QA_TEMPLATE = """You are an assistant for Disney customer experience questions.
Use the following pieces of retrieved context from Disney customer reviews to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}
Context: {context}

Answer:"""


# Metadata Extraction Template
METADATA_EXTRACTION_TEMPLATE = """You are a query parser for a Disney customer review analysis system. Extract metadata and rewrite queries for better semantic search.

Available metadata fields:
- rating: Integer (1-5) - Customer rating
- year: Integer (e.g., 2023, 2022) - Review year
- month: Integer (1-12) - Review month
- branch: String ("Disneyland", "Disney World", "Unknown") - Disney location
- reviewer_location: String - Reviewer's location
- review_id: String - Specific review ID
- original_index: Integer - Original review index


User Query: "{query}"

Extract metadata and rewrite the query following these rules:
1. Only include filters for explicitly mentioned metadata
2. If no metadata is mentioned, leave filters empty
3. For ratings, extract the specific number mentioned (1-5)
4. For years, extract 4-digit year format (e.g., 2023)
5. For months, extract month number (1-12)
6. For branches, use exact values: {branch_enum}
7. For reviewer locations, extract the mentioned location
8. For review IDs, extract specific review identifiers mentioned
9. Rewrite search_query to be more semantic and searchable
10. Set confidence as a NUMBER between 0.0 and 1.0 (0.0 = no confidence, 1.0 = high confidence)
11. Provide brief reasoning for your extraction

Return the result in the specified structured format.
{format_instructions}

"""


def get_prompt_template() -> ChatPromptTemplate:
    """Get the Disney QA prompt template.
    
    Returns:
        ChatPromptTemplate instance for Disney customer experience questions
    """
    return ChatPromptTemplate.from_template(DISNEY_QA_TEMPLATE)


def get_metadata_extraction_template() -> ChatPromptTemplate:
    """Get the metadata extraction prompt template.
    
    Returns:
        ChatPromptTemplate instance for metadata extraction
    """

    return ChatPromptTemplate.from_template(METADATA_EXTRACTION_TEMPLATE)
