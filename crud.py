from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel,Field,ConfigDict
from pymongo import MongoClient
from bson import ObjectId
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated
from motor.motor_asyncio import AsyncIOMotorClient




client=AsyncIOMotorClient()

db=client['Products']

product_collection=db['products']

app = FastAPI()



PyObjectId = Annotated[str, BeforeValidator(str)]
class Product(BaseModel):
    id: PyObjectId = Field(alias="_id", default=None)
    name:str
    price:int
    description:str
    quantity:int
    availabale:bool=False
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "name": "Jane Doe",
                "price": 2,
                "description": "Experiments, Science, and Fashion in Nanophotonics",
                "quantity": 3,
                "available":"True",
            }
        },
    )


@app.post('/students')
async def creating_product(product:Product):

    new_product=await product_collection.insert_one(product.model_dump(by_alias=True,exclude=["id"]))  

    created_product= await product_collection.find_one( {"_id": new_product.inserted_id})
    created_product["_id"] = str(created_product["_id"])

    return created_product

@app.get('/studentdetails')
async def studentlist():
    products = await product_collection.find().to_list(length=None)  
    for product in products:
        product["_id"] = str(product["_id"])  # Convert ObjectId to string
    return products