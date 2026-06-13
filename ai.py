import os, redis, json, hashlib
from google import genai
from google.genai import types

_gclient = None
_redis   = None

EMBED_MODEL = "gemini-embedding-001"
GEN_MODEL   = "gemini-2.5-flash"


def _gc():
    global _gclient
    if _gclient is None:
        _gclient = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return _gclient


def _r():
    global _redis
    if _redis is None:
        _redis = redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)
    return _redis


def get_embedding(text: str, task_type: str = "RETRIEVAL_QUERY") -> list[float]:
    key = "emb:" + hashlib.md5((task_type + text).encode()).hexdigest()
    cached = _r().get(key)
    if cached:
        return json.loads(cached)
    response = _gc().models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=768,
        ),
    )
    vec = response.embeddings[0].values
    _r().set(key, json.dumps(vec), ex=86400)
    return vec


def generate_summary(query: str, products: list[dict]) -> str:
    names  = ", ".join(p.get("name", "") for p in products[:5])
    prompt = (
        f"Un cliente busca: '{query}'.\n"
        f"Productos más relevantes: {names}.\n"
        "En una sola oración en español, da una recomendación breve y útil."
    )
    return _gc().models.generate_content(model=GEN_MODEL, contents=prompt).text.strip()


PRODUCTS = [
    {"product_id": "p001", "name": "Olla arrocera Oster 1.8L",      "category": "Electrodomésticos",   "price": 129.90, "stock":  42, "description": "Cocina arroz perfecto automáticamente, mantiene caliente."},
    {"product_id": "p002", "name": "Arroz Costeño Extra 5kg",        "category": "Abarrotes",           "price":  24.50, "stock": 200, "description": "Arroz de grano largo ideal para todo tipo de preparaciones."},
    {"product_id": "p003", "name": "Gatorade Zero Limón 500ml",      "category": "Bebidas",             "price":   4.90, "stock": 150, "description": "Bebida isotónica sin azúcar, repone electrolitos para deportistas."},
    {"product_id": "p004", "name": "Powerade Mora 600ml",            "category": "Bebidas",             "price":   3.80, "stock": 120, "description": "Bebida rehidratante con vitaminas B3, B6 y B12, sin calorías."},
    {"product_id": "p005", "name": "Agua San Luis Sin Gas 2.5L",     "category": "Bebidas",             "price":   3.20, "stock": 300, "description": "Agua de mesa purificada, sin gas y sin calorías."},
    {"product_id": "p006", "name": "Sartén antiadherente 28cm Tefal","category": "Menaje",              "price":  89.90, "stock":  30, "description": "Recubrimiento Titanium, apta para todo tipo de cocinas."},
    {"product_id": "p007", "name": "Leche Gloria Entera 1L",         "category": "Lácteos",             "price":   4.50, "stock": 400, "description": "Leche UHT entera, fuente de calcio y vitaminas A y D."},
    {"product_id": "p008", "name": "Proteína Whey Vainilla 1kg",     "category": "Nutrición deportiva", "price": 159.90, "stock":  25, "description": "Suplemento proteico para recuperación muscular post entrenamiento."},
    {"product_id": "p009", "name": "Yerba Mate Taragüi 500g",        "category": "Infusiones",          "price":  18.90, "stock":  60, "description": "Yerba mate tradicional, energizante natural con antioxidantes."},
    {"product_id": "p010", "name": "Licuadora Oster 600W",           "category": "Electrodomésticos",   "price": 119.90, "stock":  18, "description": "10 velocidades + pulso, jarra de vidrio 1.25L para smoothies."},
    {"product_id": "p011", "name": "Avena 3 Ositos 400g",            "category": "Cereales",            "price":   6.90, "stock": 180, "description": "Avena en hojuelas, rica en fibra, ideal para desayunos nutritivos."},
    {"product_id": "p012", "name": "Aceite de Oliva Borges 500ml",   "category": "Abarrotes",           "price":  22.90, "stock":  75, "description": "Aceite de oliva extra virgen, prensado en frío."},
]


def seed_products(upsert_mongo, upsert_vec, embed_fn) -> int:
    for p in PRODUCTS:
        upsert_mongo(p)
        # Documentos se indexan con RETRIEVAL_DOCUMENT (igual que el proyecto de referencia)
        vec = embed_fn(
            f"{p['name']} {p['category']} {p['description']}",
            task_type="RETRIEVAL_DOCUMENT",
        )
        upsert_vec(p["product_id"], vec)
    return len(PRODUCTS)
