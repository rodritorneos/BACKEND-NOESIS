from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from database import Base

class Usuario(Base):
    __tablename__ = "usuarios"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    
    # Relaciones
    favoritos = relationship("Favorito", back_populates="usuario", cascade="all, delete-orphan")
    visitas = relationship("Visita", back_populates="usuario", cascade="all, delete-orphan")
    puntajes = relationship("Puntaje", back_populates="usuario", cascade="all, delete-orphan")

class Favorito(Base):
    __tablename__ = "favoritos"
    
    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    clase_id = Column(String, nullable=False)
    nombre_clase = Column(String, nullable=False)
    imagen_path = Column(String, nullable=False)
    
    # Relación
    usuario = relationship("Usuario", back_populates="favoritos")

class Visita(Base):
    __tablename__ = "visitas"
    
    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    clase_id = Column(String, nullable=False)
    count = Column(Integer, default=1)
    
    # Relación
    usuario = relationship("Usuario", back_populates="visitas")

class Puntaje(Base):
    __tablename__ = "puntajes"
    
    id = Column(Integer, primary_key=True, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    puntaje_obtenido = Column(Integer, default=0)
    puntaje_total = Column(Integer, default=20)
    nivel = Column(String, default="Básico")
    
    # Relación
    usuario = relationship("Usuario", back_populates="puntajes")