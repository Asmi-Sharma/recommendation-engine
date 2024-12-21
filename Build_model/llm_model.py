from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

system = """
You are a recommendation system tasked with suggesting products based on user input.

**Input Details:**
1. **User Query:** "{user_query}" - the search term provided by the user.
2. **User History:** A detailed list of products previously purchased or interacted with by the user, including product name, category, price, discount, rating, and reviews:
{user_history}
3. **Product List:** A catalog of products available for recommendation, with details such as name, category, price, discount, rating, and description:
{product_data}

**Your Task:**
Recommend up to 5 products from the product list that best match the user query and align with the user's transactional behavior and preferences. See what kind of product and their brand are purchased by the user previously based on which recommend the products. Follow these guidelines:
- Prioritize products closely related to the query "{user_query}" and those similar to items in the user history.
- Avoid recommending products unrelated to the query or user behavior.
- If no suitable recommendations can be made, respond with "I don't know."

**Output Format:**
- Provide a list of up to 5 product_id only.
- Do not include any explanations or additional text.

"""


def get_model():
    prompt = PromptTemplate(
        template=system,
        input_variable=["user_query", "user_history",
                        "product_data", "user_query"]
    )
    llm = Ollama(model="llama3.2:3b")
    llm_chain = prompt | llm | StrOutputParser()
    return llm_chain
