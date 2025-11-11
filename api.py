from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Union
import base64
import psycopg2
import psycopg2.extras
import os
from dotenv import load_dotenv
from graph import app as langgraph_app

load_dotenv()

# Initialize FastAPI app
api = FastAPI(
    title="Product Search API",
    description="AI-powered product search with text and image support",
    version="1.0.0"
)

# CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def optional_file(image: Union[UploadFile, str, None] = File(default=None)) -> Optional[UploadFile]:
    """Handle optional file upload - converts empty strings to None."""
    if image == "" or image is None:
        return None
    if isinstance(image, str):
        return None
    return image


# Request/Response models
class SearchRequest(BaseModel):
    query: str
    image_base64: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "query": "matte black faucet from Riobel under $500",
                "image_base64": None
            }
        }


class SearchResponse(BaseModel):
    success: bool
    error_message: Optional[str] = None
    filters: Optional[str] = None
    search_results: Optional[List[str]] = None
    result_count: int = 0


def get_products_by_skus(skus: List[str]) -> List[dict]:
    """Query PostgreSQL database for products by SKU IDs."""
    if not skus:
        return []

    conn = psycopg2.connect(
        host="35.182.153.121",
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE", "product_db"),
        user=os.getenv("PGUSER", "aisearch"),
        password=os.getenv("PGPASSWORD", "SecuredPassword"),
    )

    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        query = """
            SELECT
                p.id, p.sku, p.name, b.name as brand,
                p.color, p.color_code, p.material, p.shape, p.installation_type,
                p.description, p.feature_description, p.features, p.product_gallery,
                p.main_image as image, p.rating, p.unit, p.kg, p.vol, p.memo,
                p.title, p.type, p.comment, p.price_insight, p.in_db,
                p.availability_status, p.current_count, p.order_count,
                pr.price, pr.regular_price, pr.discount, pr.discount_price, pr.last_cost,
                inv.quantity, inv.ordered, inv.delivered, inv.backorder_quantity
            FROM products p
            LEFT JOIN brands b ON p.brand_id = b.id
            LEFT JOIN product_pricing pr ON p.id = pr.product_id
            LEFT JOIN product_inventory inv ON p.id = inv.product_id
            WHERE p.sku = ANY(%s)
            ORDER BY array_position(%s, p.sku)
        """

        cursor.execute(query, (skus, skus))
        rows = cursor.fetchall()

        products = []
        for row in rows:
            product = {
                "id": row.get("id"),
                "sku": row.get("sku"),
                "name": row.get("name"),
                "brand": row.get("brand"),
                "color": row.get("color"),
                "color_code": row.get("color_code"),
                "material": row.get("material"),
                "shape": row.get("shape"),
                "installationType": row.get("installation_type") or "fixed",
                "description": row.get("description"),
                "feature_description": row.get("feature_description"),
                "features": row.get("features") or [],
                "product_gallery": row.get("product_gallery") or [],
                "image": row.get("image"),
                "rating": float(row["rating"]) if row.get("rating") else 0,
                "unit": row.get("unit") or "EA",
                "kg": float(row["kg"]) if row.get("kg") else 0,
                "vol": row.get("vol"),
                "memo": row.get("memo"),
                "title": row.get("title"),
                "type": row.get("type") or "sku",
                "comment": row.get("comment"),
                "price_insight": row.get("price_insight") or False,
                "in_db": row.get("in_db") or True,
                "status": row.get("availability_status") or "Available",
                "currentCount": row.get("current_count") or 0,
                "order": row.get("order_count") or 0,
                "price": float(row["price"]) if row.get("price") else 0,
                "regularPrice": float(row["regular_price"]) if row.get("regular_price") else 0,
                "discount": float(row["discount"]) if row.get("discount") else 0,
                "discountPrice": float(row["discount_price"]) if row.get("discount_price") else 0,
                "last_cost": float(row["last_cost"]) if row.get("last_cost") else 0,
                "quantity": row.get("quantity") or 0,
                "ordered": row.get("ordered") or 0,
                "delivered": row.get("delivered") or 0,
                "backorder_quantity": row.get("backorder_quantity") or 0
            }
            products.append(product)

        cursor.close()
        conn.close()

        return products

    except Exception as e:
        if conn:
            conn.close()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@api.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Product Search API is running",
        "version": "1.0.0"
    }


@api.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "api": "running",
            "langgraph": "running",
            "qdrant": "connected"
        }
    }


@api.post("/search", response_model=SearchResponse)
async def search_products(request: SearchRequest):
    """
    Search for products using text and/or image.

    - **query**: Text search query (optional if image provided)
    - **image_base64**: Base64 encoded image (optional)

    Returns list of matching product SKUs
    """
    try:
        # Prepare initial state
        state = {
            "user_query": request.query or "",
            "image_base64": request.image_base64,
            "guardrails_passed": False,
            "error_message": None,
            "filters": None,
            "search_results": None
        }

        # Validate input
        if not state["user_query"] and not state["image_base64"]:
            raise HTTPException(
                status_code=400,
                detail="Either query text or image must be provided"
            )

        # Execute LangGraph workflow
        result = langgraph_app.invoke(state)

        # Check for errors
        if result.get("error_message"):
            return SearchResponse(
                success=False,
                error_message=result["error_message"],
                result_count=0
            )

        # Return successful results
        search_results = result.get("search_results", [])
        return SearchResponse(
            success=True,
            filters=result.get("filters"),
            search_results=search_results,
            result_count=len(search_results)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/search/upload", response_model=SearchResponse)
async def search_with_image_upload(
    query: str = Form(default=""),
    image: UploadFile = File(default=None)
):
    """
    Search for products with file upload support.

    - **query**: Text search query (optional if image provided)
    - **image**: Image file upload (optional)

    Returns list of matching product SKUs
    """
    try:
        image_base64 = None

        # Convert uploaded image to base64
        if image:
            contents = await image.read()
            image_base64 = f"data:{image.content_type};base64,{base64.b64encode(contents).decode()}"

        # Validate input
        if not query and not image_base64:
            raise HTTPException(
                status_code=400,
                detail="Either query text or image must be provided"
            )

        # Prepare initial state
        state = {
            "user_query": query,
            "image_base64": image_base64,
            "guardrails_passed": False,
            "error_message": None,
            "filters": None,
            "search_results": None
        }

        # Execute LangGraph workflow
        result = langgraph_app.invoke(state)

        # Check for errors
        if result.get("error_message"):
            return SearchResponse(
                success=False,
                error_message=result["error_message"],
                result_count=0
            )

        # Return successful results
        search_results = result.get("search_results", [])
        return SearchResponse(
            success=True,
            filters=result.get("filters"),
            search_results=search_results,
            result_count=len(search_results)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/aisearch")
async def aisearch(
    user_query: str = Form(default=""),
    image: Optional[UploadFile] = Depends(optional_file)
):
    """
    Search products by text query and/or image.
    At least one of user_query or image must be provided.
    """
    try:
        user_query = (user_query or "").strip()

        if not user_query and not (image and hasattr(image, 'filename') and image.filename):
            raise HTTPException(
                status_code=400,
                detail="Either user_query or image must be provided"
            )

        image_base64 = None
        if image and hasattr(image, 'filename') and image.filename:
            image_bytes = await image.read()
            image_base64 = f"data:{image.content_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"

        result = langgraph_app.invoke({
            "user_query": user_query,
            "image_base64": image_base64,
            "guardrails_passed": False,
            "error_message": None,
            "filters": None,
            "search_results": None
        })

        empty_data = {
            "products": [],
            "suggested_products": [],
            "suggested_products_pagination": {
                "page": 1,
                "pages": 0,
                "per_page": 20,
                "total": 0
            }
        }

        if result.get("error_message"):
            return {"success": False, "data": empty_data, "error": result["error_message"]}

        sku_ids = result.get("search_results", [])
        if not sku_ids:
            return {"success": True, "data": empty_data}

        products = get_products_by_skus(sku_ids)
        return {
            "success": True,
            "data": {
                "products": products,
                "suggested_products": [],
                "suggested_products_pagination": {
                    "page": 1,
                    "pages": 0,
                    "per_page": 20,
                    "total": 0
                }
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)
