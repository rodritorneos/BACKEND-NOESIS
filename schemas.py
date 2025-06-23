from pydantic import BaseModel
from typing import List, Optional, Any

# Schemas para Usuario
class UsuarioRegistro(BaseModel):
    email: str
    password: str

class UsuarioLogin(BaseModel):
    email: str
    password: str

class UsuarioResponse(BaseModel):
    email: str
    password: str

class UsuarioInfo(BaseModel):
    email: str

class UsuarioUpdateProfile(BaseModel):
    email: str

class UsuarioChangePassword(BaseModel):
    new_password: str

# Schemas para Favorito
class FavoritoRequest(BaseModel):
    clase_id: str
    nombre_clase: str
    imagen_path: str

class FavoritoResponse(BaseModel):
    clase_id: str
    nombre_clase: str
    imagen_path: str

class FavoritosUsuarioResponse(BaseModel):
    email: str
    favoritos: List[FavoritoResponse]
    total: int

# Schemas para Visita
class VisitaRequest(BaseModel):
    clase_id: str

class VisitaResponse(BaseModel):
    clase_id: str
    count: int

class VisitasUsuarioResponse(BaseModel):
    email: str
    visitas: List[VisitaResponse]
    total_visitas: int

# Schemas para Puntaje
class PuntajeRequest(BaseModel):
    puntaje_obtenido: int
    puntaje_total: int
    nivel: str

class PuntajeResponse(BaseModel):
    email: str
    puntaje_obtenido: int
    puntaje_total: int
    nivel: str

class PuntajeUpdateResponse(BaseModel):
    message: str
    data: dict

# Schemas para respuestas generales
class MessageResponse(BaseModel):
    message: str

class RegistroResponse(BaseModel):
    message: str
    email: str

class LoginResponse(BaseModel):
    message: str
    email: str

class FavoritoAddResponse(BaseModel):
    message: str
    favorito: FavoritoResponse

class HealthResponse(BaseModel):
    status: str
    database: str
    usuarios_registrados: int
    total_favoritos: int
    total_visitas: int
    total_puntajes: int
    database_ok: bool

class RootResponse(BaseModel):
    message: str
    version: str
    database: str
    endpoints: dict

# Schemas para ML Model
class ModelPredictRequest(BaseModel):
    puntaje_obtenido: int
    puntaje_total: int

class ModelPredictResponse(BaseModel):
    nivel_predicho: str
    porcentaje: float
    puntaje_obtenido: int
    puntaje_total: int
    probabilidades: Optional[dict] = None
    confianza: Optional[float] = None

class ModelStatsResponse(BaseModel):
    modelo_cargado: bool
    tipo_modelo: Optional[str] = None
    estadisticas: Any

class ModelPredictUserRequest(BaseModel):
    """Request para predicción basada en usuario existente"""
    pass  # No necesita campos, usa el email de la URL

class ModelPredictUserResponse(BaseModel):
    """Response para predicción de usuario con datos completos"""
    email: str
    nivel_predicho: str
    nivel_actual_guardado: str
    porcentaje: float
    puntaje_obtenido: int
    puntaje_total: int
    probabilidades: Optional[dict] = None
    confianza: Optional[float] = None
    es_prediccion_mejor: bool

class ModelUserStatsResponse(BaseModel):
    """Response para estadísticas del usuario y modelo"""
    email: str
    puntaje_actual: dict
    prediccion_actual: dict
    estadisticas_modelo: dict
    recomendacion: str