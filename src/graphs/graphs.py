from langgraph.graph import StateGraph, END, START
from utils.llm import LLMModel
from tools.tools import retriever_tool, search_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from src.graphs._schema import RecipeBotState, IsItRecipeRelevant, GradeDocuments
from langgraph.checkpoint.memory import MemorySaver

llm_instance = LLMModel()
llm = llm_instance.get_model()


def is_question_recipe_related():
    """Get whether the question is food or recipe related."""
    # LLM with function call
    structured_llm_checker = llm.with_structured_output(IsItRecipeRelevant)
    # Prompt
    system = """
    Your job is to act as a strict binary classifier.
    You will receive a user question and must respond with only one word: 'yes' or 'no'.
    Do not provide any other text, punctuation, or capitalization.
    'yes' if the question is related to food recipes, cooking, ingredients, or flavors.
    'no' otherwise.
    """
    relevance_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: {question}")
        ]
    )

    relevance_checker = relevance_prompt | structured_llm_checker

    return relevance_checker

def should_generate_or_retrieve(state: RecipeBotState) -> str:
    """
        Determine whether the question is related to food recipe

        Args:
            state(dict): current state of the graph

        Returns:
            str: Binary decision for next node to call
    """

    recipe_relevant = state["recipe_relevant"].lower()

    if 'yes' in recipe_relevant:
        print("go to retrieve")
        return "retrieve"
    else:
        print("go to generate")
        return "generate"

async def grade_question(state: RecipeBotState) -> RecipeBotState:
    """
        Determine whether the question is related to food recipe

        Args:
            state(dict): current state of the graph

        Returns:
            state (dict): Updates recipe_relevant key with question graded for recipe relevance
    """

    question = state["question"]
    relevance_checker = is_question_recipe_related()
    score = await relevance_checker.ainvoke({"question": question})

    print(f"Question relevance graded as: {score.binary_score.lower()}")

    grade = 'yes' if 'yes' in score.binary_score.lower() else 'no'

    print(f"Question relevance graded as: {grade}")

    return {"question": question, "recipe_relevant": grade}
    

def doc_relevance_grader():
    """Get the document relevance grade."""
    # LLM with function call
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    # Prompt
    system = """
            You are a grader assessing the relevance of a retrieved document to a user's question about food.

            Your task is to determine if the document contains information that can help answer the question.

            Instructions:

            1.  Grade the document as 'yes' if it contains keywords, concepts, or semantic meaning related to the user's question.
            2.  A document is relevant if it provides a recipe, ingredients, or a general food idea that matches the user's query.
            3.  If the user asks for a specific food type, a document mentioning related ingredients or dishes is relevant. For example, if the user asks for a "pasta recipe without cheese," a document mentioning "pasta," "tomato sauce," or a recipe that does not include cheese is relevant.
            4.  A document is relevant if it contains information about a cuisine or a specific region that is a subset of a broader geographical area mentioned in the user's question. For example, if the user asks for a "European dish" and the document mentions "Italian food," it is relevant.
            5.  Even if a word isn't explicitly mentioned in the document, consider synonyms, subsets, or related terms. For example, "Asian" could relate to "Thai," "Chinese," "Korean," "Japanese," etc.
            6.  Grade the document 'no' only if it is completely unrelated to the user's food-related question.

            Output:
            Provide a single binary score: 'yes' or 'no'.
        """
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader

    return retrieval_grader

async def grade_documents(state: RecipeBotState) -> RecipeBotState:
    """
        Document grading to determine whether a document is relevant to a user's question. 

        Args:
            state(dict): current state of the graph

        Returns:
            state (dict): Updates web_search key graded for documents relevance against user question
    """
    print("Grading documents...")
    question = state["question"]
    documents = state["documents"]

    web_search = "no"
    documents_relevant = "no"

    retrieval_grader = doc_relevance_grader()
    # First, ensure documents is iterable and elements are strings
    if isinstance(documents, str):
        # documents is already a string, no need to join
        document_text = documents
    elif hasattr(documents, '__iter__'):
        # documents is iterable, convert all elements to strings and join with space
        document_text = " ".join(str(doc) for doc in documents)
    else:
        # documents is neither string nor iterable, convert to string directly
        document_text = str(documents)

    print("Document text for grading:", document_text)

    score = await retrieval_grader.ainvoke({"question": question, "document": document_text})

    # score = await retrieval_grader.ainvoke({"question": question, "document": "".join(documents)})
    grade = score.binary_score
    print(f"Document relevance graded as: {grade}")
    if grade == 'yes':
        print("documents are relevant")
        documents_relevant = "yes"
        web_search = "no"
    else:
        print("documents are NOT relevant")
        documents_relevant = "no"
        web_search = 'yes'
    return {"documents": documents, "question": question, "web_search": web_search, "documents_relevant": documents_relevant}

async def retrieve_documents(state: RecipeBotState) -> RecipeBotState:
    """Retrieve documents based on the question."""
    print("Retrieving documents...")
    question = state["question"]
    documents = await retriever_tool.ainvoke(question)
    
    if not documents:
        return {"documents": [], "question": question, "web_search": "yes"}

    print(documents)
    
    return {"documents": documents, "question": question, "web_search": "no"}

def decide_to_generate(state):
    """
        Determine whether to generate an answer or regenerate question to perform web search

        Args:
            state (dict): current state of the graph

        Returns:
            str: Binary decision for next node to call
    """
    state["question"]
    web_search = state.get("web_search", "no")
    state["documents"]

    if web_search == "yes":
        print("go to web search...")
        return "web_search"
    else:
        # Documents are relevant
        print("go to generate...")
        return "generate"
    
async def web_search(state):
    """
        Web search based on question

        Args:
            state (dict): current state of the graph

        Returns:
            state (dict): Updates documents key with appended web results
    """

    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = await search_tool.ainvoke({"query": question})

    # Extract and concatenate content from all results
    if isinstance(docs, list) and all("content" in d for d in docs):
        web_results = "\n\n".join(d["content"] for d in docs)
    else:
        web_results = ""

    web_results = Document(page_content=web_results)
    documents = web_results.page_content

    return {"documents": documents, "question": question}

async def generate(state: RecipeBotState) -> RecipeBotState:
    print("Generating answer...")
    system = """   
        You are my expert personal assistant. Your main task is to generate a detailed recipe from the provided context.

        **Instructions:**
        1.  **First, determine the source of the recipe.**
            * If **Documents Relevant** is 'yes', the recipe is from "Tann's personal recipes."
            * If **Documents Relevant** is 'no' AND **Web Search** is 'yes', the recipe is from "outside sources."
            * In all other cases, do not state a source.
        2.  **Next, generate the recipe.**
            * Use the provided `Context` to create a detailed recipe.
            * Be sure to include all ingredients and steps. **DO NOT SKIP ANY INFORMATION**.
        3.  **Finally, structure your response.**
            * Start your response with the source statement from step 1, if applicable.
            * Answer in a casual, caring tone, as if you're teaching your younger brother.

        **Input:**
        User question: {question}
        Context: {context}
        Web search: {web_search}
        Documents Relevant: {documents_relevant}
        """
    generate_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {context} \n\n User question: {question} \n\n From web search: {web_search} \n\n Recipe relevant: {recipe_relevant} \n\n Documents relevant: {documents_relevant} \n\n Generate answer")
        ]
    )
    question = state["question"]
    documents = state.get("documents", [])
    web_search = state.get("web_search", "")
    recipe_relevant = state.get("recipe_relevant", "no")
    documents_relevant = state.get("documents_relevant", "no")

    if isinstance(documents, list) and documents:
        context_string = "\n\n".join(doc.page_content for doc in documents if isinstance(doc, Document))
    elif isinstance(documents, str):
        context_string = documents
    else:
        context_string = ""

    rag_chain = generate_prompt | llm

    generation = await rag_chain.ainvoke({"context": context_string, "question": question, "web_search": web_search, "recipe_relevant": recipe_relevant, "documents_relevant": documents_relevant})

    return {"documents": documents, "question": question, "generation": generation}

def create_rag_graph():
    graph = StateGraph(RecipeBotState, checkpoint=MemorySaver())

    graph.add_node("recipe_relevancy", grade_question)
    graph.add_node("retrieve", retrieve_documents)
    graph.add_node("grade", grade_documents)
    graph.add_node("generate", generate)
    graph.add_node("web_search", web_search)

    graph.add_edge(START, "recipe_relevancy")

    graph.add_conditional_edges(
        "recipe_relevancy",
        should_generate_or_retrieve,
        {
            "retrieve": "retrieve",
            "generate": "generate"
        }
    )

    graph.add_edge("retrieve", "grade")

    # Grade the documents. If document is relevant, go straight to generate. If not, go to web search
    graph.add_conditional_edges(
        "grade",
        decide_to_generate,
        {
            "web_search": "web_search",
            "generate": "generate"
        }
    )

    graph.add_edge("web_search", "generate")
    graph.add_edge("generate", END)
    
    return graph.compile()

# Used for local testing, leaving it here for now
async def get_response_from_rag(question: str) -> str:
    """Get response from RAG graph based on user question."""
    rag_graph = create_rag_graph()
    response = await rag_graph.ainvoke({"question": question})
    print(response["generation"].content)
    return response["generation"].content

app = create_rag_graph()
