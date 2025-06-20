from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Leer la URL de conexión desde la variable de entorno personalizada
DATABASE_URL = os.getenv("NOESIS_URL")

# Validar que esté definida
if not DATABASE_URL:
    raise ValueError("La variable de entorno 'NOESIS_URL' no está definida.")

# Crear el motor de conexión a PostgreSQL (ya no se necesita connect_args)
engine = create_engine(DATABASE_URL)

# Crear la sesión para interactuar con la base de datos
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base declarativa para los modelos
Base = declarative_base()

# Función para obtener una sesión de base de datos
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()