from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from Fetch_data import similarProduct, userTransactionData
from Build_model import llm_model

router = APIRouter()


class UserInput(BaseModel):
    user_id: str
    query: str


@router.post("/recommend-product")
def recommend_product(user_id, query):
    try:
        fields_name = UserInput(user_id=user_id, query=query)
        product_list = similarProduct.run_query_vectordb(fields_name.query)
        user_history = userTransactionData.run_query_postgresql(
            fields_name.user_id)
        llm_chain = llm_model.get_model()
        response = llm_chain.invoke({
            'user_query': fields_name.query,
            'user_history': user_history,
            'product_data': product_list,
            'user_query': fields_name.query
        })
        return response

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid Input Format or Inputs \n {e}")
