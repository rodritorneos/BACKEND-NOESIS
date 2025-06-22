from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List

# Agregar estos imports al inicio
from model_predict import predictor
from schemas import ModelPredictRequest, ModelPredictResponse, ModelStatsResponse

from schemas import ModelPredictUserRequest, ModelPredictUserResponse, ModelUserStatsResponse

from database import engine, get_db
from models import Usuario, Favorito, Visita, Puntaje, Base
from schemas import (
    UsuarioRegistro, UsuarioLogin, UsuarioResponse, UsuarioInfo,
    FavoritoRequest, FavoritoResponse, FavoritosUsuarioResponse, FavoritoAddResponse,
    VisitaRequest, VisitaResponse, VisitasUsuarioResponse,
    PuntajeRequest, PuntajeResponse, PuntajeUpdateResponse,
    MessageResponse, RegistroResponse, LoginResponse,
    HealthResponse, RootResponse
)

# Agregar logger
import logging
logger = logging.getLogger(__name__)

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

# Funciones auxiliares
def get_user_by_email(db: Session, email: str):
    return db.query(Usuario).filter(Usuario.email == email).first()

def create_user(db: Session, email: str, password: str):
    db_user = Usuario(email=email, password=password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Crear puntaje inicial
    db_puntaje = Puntaje(
        usuario_id=db_user.id,
        puntaje_obtenido=0,
        puntaje_total=20,
        nivel="Básico"
    )
    db.add(db_puntaje)
    db.commit()
    
    return db_user

def _compare_levels(nivel_predicho: str, nivel_guardado: str) -> bool:
    """Comparar si el nivel predicho es mejor que el guardado"""
    level_hierarchy = {"Básico": 1, "Intermedio": 2, "Avanzado": 3}
    return level_hierarchy.get(nivel_predicho, 1) > level_hierarchy.get(nivel_guardado, 1)

def _generate_user_recommendation(puntaje: Puntaje, prediction: dict) -> str:
    """Generar recomendación personalizada para el usuario"""
    nivel_predicho = prediction['nivel_predicho']
    nivel_guardado = puntaje.nivel
    porcentaje = prediction['porcentaje']
    confianza = prediction.get('confianza', 0)
    
    if nivel_predicho == nivel_guardado:
        if porcentaje >= 80:
            return f"¡Excelente! Tu nivel {nivel_predicho} está bien consolidado. Considera intentar contenido más avanzado."
        elif porcentaje >= 60:
            return f"Tu nivel {nivel_predicho} es correcto. Practica más para fortalecer tus conocimientos."
        else:
            return f"Tu nivel {nivel_predicho} necesita refuerzo. Te recomendamos más práctica en las áreas básicas."
    
    elif _compare_levels(nivel_predicho, nivel_guardado):
        return f"¡Felicidades! El modelo predice que has avanzado al nivel {nivel_predicho}. Continúa practicando para consolidar este progreso."
    
    else:
        return f"El modelo sugiere revisar contenido de nivel {nivel_predicho}. Esto puede ser temporal, sigue practicando."


# Endpoints de usuarios
@app.get("/usuarios", response_model=List[UsuarioResponse])
async def get_usuarios(db: Session = Depends(get_db)):
    """Obtener todos los usuarios (solo email y password para compatibilidad)"""
    usuarios = db.query(Usuario).all()
    # Manteniendo compatibilidad con el frontend actual
    return [{"email": user.email, "password": user.password} for user in usuarios]

@app.post("/usuarios/registro", response_model=RegistroResponse)
async def registrar_usuario(usuario: UsuarioRegistro, db: Session = Depends(get_db)):
    """Registrar un nuevo usuario"""
    # Verificar si el email ya existe
    if get_user_by_email(db, usuario.email):
        raise HTTPException(status_code=400, detail="El email ya está registrado")
    
    # Crear nuevo usuario
    nuevo_usuario = create_user(db, usuario.email, usuario.password)
    
    return {"message": "Usuario registrado exitosamente", "email": usuario.email}

@app.post("/usuarios/login", response_model=LoginResponse)
async def login_usuario(usuario: UsuarioLogin, db: Session = Depends(get_db)):
    """Autenticar usuario"""
    usuario_encontrado = get_user_by_email(db, usuario.email)
    
    if not usuario_encontrado:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    if usuario_encontrado.password != usuario.password:
        raise HTTPException(status_code=401, detail="Contraseña incorrecta")
    
    return {"message": "Login exitoso", "email": usuario.email}

@app.get("/usuarios/{email}", response_model=UsuarioInfo)
async def obtener_usuario(email: str, db: Session = Depends(get_db)):
    """Obtener información de un usuario específico"""
    usuario = get_user_by_email(db, email)
    
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    return {"email": usuario.email}

@app.delete("/usuarios/{email}", response_model=MessageResponse)
async def eliminar_usuario(email: str, db: Session = Depends(get_db)):
    """Eliminar un usuario y todos sus datos relacionados"""
    usuario = get_user_by_email(db, email)
    
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    # SQLAlchemy eliminará automáticamente los datos relacionados gracias a cascade="all, delete-orphan"
    db.delete(usuario)
    db.commit()
    
    return {"message": "Usuario, favoritos, visitas y puntajes eliminados exitosamente"}

# Endpoints de favoritos
@app.post("/usuarios/{email}/favoritos", response_model=FavoritoAddResponse)
async def agregar_favorito(email: str, favorito: FavoritoRequest, db: Session = Depends(get_db)):
    """Agregar una clase a favoritos"""
    usuario = get_user_by_email(db, email)
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    # Verificar si ya está en favoritos
    favorito_existente = db.query(Favorito).filter(
        Favorito.usuario_id == usuario.id,
        Favorito.clase_id == favorito.clase_id
    ).first()
    
    if favorito_existente:
        raise HTTPException(status_code=400, detail="La clase ya está en favoritos")
    
    # Crear nuevo favorito
    nuevo_favorito = Favorito(
        usuario_id=usuario.id,
        clase_id=favorito.clase_id,
        nombre_clase=favorito.nombre_clase,
        imagen_path=favorito.imagen_path
    )
    
    db.add(nuevo_favorito)
    db.commit()
    db.refresh(nuevo_favorito)
    
    return {
        "message": "Favorito agregado exitosamente",
        "favorito": {
            "clase_id": nuevo_favorito.clase_id,
            "nombre_clase": nuevo_favorito.nombre_clase,
            "imagen_path": nuevo_favorito.imagen_path
        }
    }

@app.delete("/usuarios/{email}/favoritos/{clase_id}", response_model=MessageResponse)
async def remover_favorito(email: str, clase_id: str, db: Session = Depends(get_db)):
    """Remover una clase de favoritos"""
    usuario = get_user_by_email(db, email)
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    favorito = db.query(Favorito).filter(
        Favorito.usuario_id == usuario.id,
        Favorito.clase_id == clase_id
    ).first()
    
    if not favorito:
        raise HTTPException(status_code=404, detail="Favorito no encontrado")
    
    db.delete(favorito)
    db.commit()
    
    return {"message": "Favorito removido exitosamente"}

@app.get("/usuarios/{email}/favoritos", response_model=FavoritosUsuarioResponse)
async def obtener_favoritos_usuario(email: str, db: Session = Depends(get_db)):
    """Obtener los favoritos de un usuario"""
    usuario = get_user_by_email(db, email)
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    favoritos = db.query(Favorito).filter(Favorito.usuario_id == usuario.id).all()
    
    favoritos_list = [
        {
            "clase_id": fav.clase_id,
            "nombre_clase": fav.nombre_clase,
            "imagen_path": fav.imagen_path
        }
        for fav in favoritos
    ]
    
    return {
        "email": email,
        "favoritos": favoritos_list,
        "total": len(favoritos_list)
    }

@app.put("/usuarios/{email}/favoritos/{clase_id}", response_model=MessageResponse)
async def actualizar_favorito(email: str, clase_id: str, favorito_actualizado: FavoritoRequest, db: Session = Depends(get_db)):
    """Actualizar información de un favorito"""
    usuario = get_user_by_email(db, email)
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    favorito = db.query(Favorito).filter(
        Favorito.usuario_id == usuario.id,
        Favorito.clase_id == clase_id
    ).first()
    
    if not favorito:
        raise HTTPException(status_code=404, detail="Favorito no encontrado")
    
    favorito.clase_id = favorito_actualizado.clase_id
    favorito.nombre_clase = favorito_actualizado.nombre_clase
    favorito.imagen_path = favorito_actualizado.imagen_path
    
    db.commit()
    
    return {"message": "Favorito actualizado exitosamente"}

# Endpoints de visitas
@app.post("/usuarios/{email}/visitas", response_model=MessageResponse)
async def registrar_visita(email: str, visita: VisitaRequest, db: Session = Depends(get_db)):
    """Registrar una visita a una clase"""
    usuario = get_user_by_email(db, email)
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    # Buscar si ya existe una visita para esta clase
    visita_existente = db.query(Visita).filter(
        Visita.usuario_id == usuario.id,
        Visita.clase_id == visita.clase_id
    ).first()
    
    if visita_existente:
        # Incrementar contador
        visita_existente.count += 1
        db.commit()
    else:
        # Crear nueva visita
        nueva_visita = Visita(
            usuario_id=usuario.id,
            clase_id=visita.clase_id,
            count=1
        )
        db.add(nueva_visita)
        db.commit()
    
    return {"message": "Visita registrada exitosamente"}

@app.get("/usuarios/{email}/visitas", response_model=VisitasUsuarioResponse)
async def obtener_visitas_usuario(email: str, db: Session = Depends(get_db)):
    """Obtener las visitas de un usuario"""
    usuario = get_user_by_email(db, email)
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    visitas = db.query(Visita).filter(Visita.usuario_id == usuario.id).all()
    
    visitas_list = [
        {
            "clase_id": visita.clase_id,
            "count": visita.count
        }
        for visita in visitas
    ]
    
    total_visitas = sum(visita.count for visita in visitas)
    
    return {
        "email": email,
        "visitas": visitas_list,
        "total_visitas": total_visitas
    }

# Endpoints de puntajes
@app.get("/usuarios/{email}/puntajes", response_model=PuntajeResponse)
async def obtener_puntajes_usuario(email: str, db: Session = Depends(get_db)):
    """Obtener los puntajes de un usuario"""
    usuario = get_user_by_email(db, email)
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    puntaje = db.query(Puntaje).filter(Puntaje.usuario_id == usuario.id).first()
    
    if not puntaje:
        # Crear puntaje inicial si no existe
        puntaje = Puntaje(
            usuario_id=usuario.id,
            puntaje_obtenido=0,
            puntaje_total=20,
            nivel="Básico"
        )
        db.add(puntaje)
        db.commit()
        db.refresh(puntaje)
    
    return {
        "email": email,
        "puntaje_obtenido": puntaje.puntaje_obtenido,
        "puntaje_total": puntaje.puntaje_total,
        "nivel": puntaje.nivel
    }

@app.post("/usuarios/{email}/puntajes", response_model=PuntajeUpdateResponse)
async def actualizar_puntajes_usuario(email: str, puntaje_request: PuntajeRequest, db: Session = Depends(get_db)):
    """Actualizar los puntajes de un usuario solo si es mejor que el anterior, con predicción ML automática"""
    usuario = get_user_by_email(db, email)
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    puntaje = db.query(Puntaje).filter(Puntaje.usuario_id == usuario.id).first()
    is_new_best = False
    nivel_usado = puntaje_request.nivel  # Usar el nivel enviado por defecto
    
    # Intentar obtener predicción ML para mejorar el nivel
    try:
        ml_prediction = predictor.predict_level(
            score_obtained=puntaje_request.puntaje_obtenido,
            total_score=puntaje_request.puntaje_total
        )
        nivel_ml = ml_prediction['nivel_predicho']
        
        # Usar el nivel ML si tiene alta confianza (>0.7) o si es mejor que el enviado
        confianza = ml_prediction.get('confianza', 0)
        if confianza > 0.7 or _compare_levels(nivel_ml, puntaje_request.nivel):
            nivel_usado = nivel_ml
            
    except Exception as e:
        logger.warning(f"⚠️ Error en predicción ML, usando nivel manual: {e}")
        # Continuar con el nivel enviado en la request
    
    if not puntaje:
        # Crear nuevo puntaje
        puntaje = Puntaje(
            usuario_id=usuario.id,
            puntaje_obtenido=puntaje_request.puntaje_obtenido,
            puntaje_total=puntaje_request.puntaje_total,
            nivel=nivel_usado
        )
        db.add(puntaje)
        is_new_best = True
    else:
        # Comparar porcentajes
        porcentaje_actual = (puntaje.puntaje_obtenido / puntaje.puntaje_total) * 100
        porcentaje_nuevo = (puntaje_request.puntaje_obtenido / puntaje_request.puntaje_total) * 100
        
        if porcentaje_nuevo > porcentaje_actual:
            puntaje.puntaje_obtenido = puntaje_request.puntaje_obtenido
            puntaje.puntaje_total = puntaje_request.puntaje_total
            puntaje.nivel = nivel_usado
            is_new_best = True
    
    db.commit()
    
    return {
        "message": "Puntaje procesado exitosamente" + (" con ML" if nivel_usado != puntaje_request.nivel else ""),
        "data": {
            "is_new_best": is_new_best,
            "puntaje_obtenido": puntaje_request.puntaje_obtenido,
            "puntaje_total": puntaje_request.puntaje_total,
            "nivel": nivel_usado,
            "nivel_original": puntaje_request.nivel,
            "ml_enhanced": nivel_usado != puntaje_request.nivel
        }
    }

# Agregar estas rutas antes del endpoint /health
@app.post("/modelo/predict", response_model=ModelPredictResponse)
async def predict_english_level(request: ModelPredictRequest):
    """Predecir nivel de inglés basado en puntaje del quiz"""
    try:
        result = predictor.predict_level(
            score_obtained=request.puntaje_obtenido,
            total_score=request.puntaje_total
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.get("/modelo/stats", response_model=ModelStatsResponse)
async def get_model_stats():
    """Obtener estadísticas del modelo ML"""
    try:
        stats = predictor.get_model_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}")


@app.get("/usuarios/{email}/puntajes/modelo/predict", response_model=ModelPredictUserResponse)
async def predict_user_english_level(email: str, db: Session = Depends(get_db)):
    """Predecir nivel de inglés para un usuario específico usando su puntaje guardado"""
    # Verificar que el usuario existe
    usuario = get_user_by_email(db, email)
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    # Obtener puntaje del usuario
    puntaje = db.query(Puntaje).filter(Puntaje.usuario_id == usuario.id).first()
    if not puntaje:
        raise HTTPException(status_code=404, detail="No se encontraron puntajes para este usuario")
    
    try:
        # Realizar predicción usando el modelo
        prediction_result = predictor.predict_level(
            score_obtained=puntaje.puntaje_obtenido,
            total_score=puntaje.puntaje_total
        )
        
        # Comparar nivel predicho vs nivel guardado
        nivel_predicho = prediction_result['nivel_predicho']
        nivel_guardado = puntaje.nivel
        es_mejor = _compare_levels(nivel_predicho, nivel_guardado)
        
        return {
            "email": email,
            "nivel_predicho": nivel_predicho,
            "nivel_actual_guardado": nivel_guardado,
            "porcentaje": prediction_result['porcentaje'],
            "puntaje_obtenido": puntaje.puntaje_obtenido,
            "puntaje_total": puntaje.puntaje_total,
            "probabilidades": prediction_result.get('probabilidades'),
            "confianza": prediction_result.get('confianza'),
            "es_prediccion_mejor": es_mejor
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.get("/usuarios/{email}/puntajes/modelo/stats", response_model=ModelUserStatsResponse)
async def get_user_model_stats(email: str, db: Session = Depends(get_db)):
    """Obtener estadísticas del modelo y recomendaciones para un usuario específico"""
    # Verificar que el usuario existe
    usuario = get_user_by_email(db, email)
    if not usuario:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    
    # Obtener puntaje del usuario
    puntaje = db.query(Puntaje).filter(Puntaje.usuario_id == usuario.id).first()
    if not puntaje:
        raise HTTPException(status_code=404, detail="No se encontraron puntajes para este usuario")
    
    try:
        # Obtener predicción actual
        prediction_result = predictor.predict_level(
            score_obtained=puntaje.puntaje_obtenido,
            total_score=puntaje.puntaje_total
        )
        
        # Obtener estadísticas del modelo
        model_stats = predictor.get_model_stats()
        
        # Generar recomendación personalizada
        recomendacion = _generate_user_recommendation(puntaje, prediction_result)
        
        return {
            "email": email,
            "puntaje_actual": {
                "puntaje_obtenido": puntaje.puntaje_obtenido,
                "puntaje_total": puntaje.puntaje_total,
                "nivel_guardado": puntaje.nivel,
                "porcentaje": round((puntaje.puntaje_obtenido / puntaje.puntaje_total) * 100, 2)
            },
            "prediccion_actual": {
                "nivel_predicho": prediction_result['nivel_predicho'],
                "confianza": prediction_result.get('confianza', 0),
                "probabilidades": prediction_result.get('probabilidades', {})
            },
            "estadisticas_modelo": model_stats,
            "recomendacion": recomendacion
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}")



# Endpoints de información general
@app.get("/", response_model=RootResponse)
async def root():
    """Endpoint raíz de la API"""
    return {
        "message": "API de usuarios, favoritos, visitas y puntajes",
        "version": "2.0.0",
        "database": "PostgreSQL (Render)",
        "endpoints": {
            "usuarios": "/usuarios/{email}",
            "registro": "/usuarios/registro",
            "login": "/usuarios/login",
            "favoritos": "/usuarios/{email}/favoritos",
            "visitas": "/usuarios/{email}/visitas",
            "puntajes": "/usuarios/{email}/puntajes"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """Verificar el estado de la API"""
    try:
        usuarios_count = db.query(Usuario).count()
        favoritos_count = db.query(Favorito).count()
        visitas_count = db.query(Visita).count()
        puntajes_count = db.query(Puntaje).count()
        
        return {
            "status": "healthy",
            "database": "PostgreSQL (Render)",
            "usuarios_registrados": usuarios_count,
            "total_favoritos": favoritos_count,
            "total_visitas": visitas_count,
            "total_puntajes": puntajes_count,
            "database_ok": True
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "database_ok": False
        }

@app.get("/modelo/stats/training", response_model=dict)
async def get_model_training_stats():
    """Obtener todas las métricas de entrenamiento del modelo"""
    try:
        stats = predictor.get_model_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo métricas de entrenamiento: {str(e)}")