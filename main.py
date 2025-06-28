from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database.database import engine
from models.models import Base
import logging

# Crear las tablas en PostgreSQL si no existen
Base.metadata.create_all(bind=engine)

# Crear la aplicación FastAPI
app = FastAPI(title="API Usuarios, Favoritos, Visitas y Puntajes Noesis", version="2.0.0")

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Agregar logger (opcional pero recomendable)
logger = logging.getLogger(__name__)

# Registrar rutas desde el módulo controller.routes
from controller.routes import register_routes
register_routes(app)