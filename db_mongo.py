import os
from pymongo import MongoClient

_col = None


def _get_col():
    global _col
    if _col is None:
        client = MongoClient(os.getenv("MONGO_URL"))
        db = client[os.getenv("MONGO_DB", "smartsearch")]
        _col = db["products"]
        _col.create_index("product_id", unique=True)
    return _col


def init_mongo():
    _get_col()


def upsert_product(product: dict):
    _get_col().update_one(
        {"product_id": product["product_id"]},
        {"$set": product},
        upsert=True,
    )


def get_product_by_id(product_id: str) -> dict | None:
    return _get_col().find_one({"product_id": product_id}, {"_id": 0})
