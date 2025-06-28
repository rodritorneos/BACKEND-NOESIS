from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database.database import engine
from models.models import Base
from controller.routes import register_routes
import logging

# Crear las tablas en PostgreSQL si no existen
Base.metadata.create_all(bind=engine)

# Crear la aplicación FastAPI
app = FastAPI(
    title="NOESIS",
    description="Sistema para gestionar usuarios, puntajes, favoritos y predicciones con Machine Learning",
    version="2.0.0"
)

# Habilitar CORS (puedes ajustar origins si es necesario)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambiar por origen específico en producción (frontend) Ejemplo "https://mi-frontend.onrender.com
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Agregar logger
logger = logging.getLogger(__name__)

# Registrar rutas desde el módulo controller.routes
register_routes(app)