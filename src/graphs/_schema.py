from typing import TypedDict
from pydantic import BaseModel, Field

class RecipeBotState(TypedDict):
    """
        Represent state of the graph

        Attributes:
            question: question
            generation: LLM generation
            web_search: whether to add search
            documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: list[str]
    recipe_relevant: str
    documents_relevant: str

class IsItRecipeRelevant(BaseModel):
    """Binary score for relevance check on food recipes related question"""

    binary_score: str = Field(
        description="Is the question related to food recommendations, food recipes, cooking, ingredients, flavors, or meal preparation? 'yes' or 'no'?"
    )

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieve documents"""

    binary_score: str = Field(
        description="Documents are relevant to the question. 'yes' or 'no'?"
    )

__all__ = ["RecipeBotState", "IsItRecipeRelevant", "GradeDocuments"]