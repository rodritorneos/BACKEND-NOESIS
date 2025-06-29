from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List

# Predictor del modelo
from service.model_predict import predictor

# Schemas del modelo de predicción
from view.schemas import (
    ModelPredictRequest, ModelPredictResponse, ModelStatsResponse,
    ModelPredictUserRequest, ModelPredictUserResponse, ModelUserStatsResponse
)

# Base de datos y modelos
from database.database import engine, get_db
from models.models import Usuario, Favorito, Visita, Puntaje, Base, Docente

# Schemas generales (usuario, docente, etc.)
from view.schemas import (
    UsuarioRegistro, UsuarioLogin, UsuarioResponse, UsuarioInfo,
    UsuarioUpdateProfile, UsuarioChangePassword,
    FavoritoRequest, FavoritoResponse, FavoritosUsuarioResponse, FavoritoAddResponse,
    VisitaRequest, VisitaResponse, VisitasUsuarioResponse,
    PuntajeRequest, PuntajeResponse, PuntajeUpdateResponse,
    MessageResponse, RegistroResponse, LoginResponse,
    HealthResponse, RootResponse,
    DocenteRegistro, DocenteLogin, DocenteResponse, DocenteInfo
)

# Utilidades adicionales
from sqlalchemy import func, case, desc
from datetime import datetime, timedelta

# Logging
import logging
logger = logging.getLogger(__name__)

def register_routes(app: FastAPI):

    def get_user_by_email(db: Session, email: str):
        return db.query(Usuario).filter(Usuario.email == email).first()

    def get_user_by_username(db: Session, username: str):
        return db.query(Usuario).filter(Usuario.username == username).first()

    def create_user(db: Session, username: str, email: str, password: str):
        db_user = Usuario(username=username, email=email, password=password)
        db.add(db_user)
        db.commit()
        db.refresh(db_user)

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
        level_hierarchy = {"Básico": 1, "Intermedio": 2, "Avanzado": 3}
        return level_hierarchy.get(nivel_predicho, 1) > level_hierarchy.get(nivel_guardado, 1)

    def _generate_user_recommendation(puntaje: Puntaje, prediction: dict) -> str:
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

    def calculate_general_stats(db: Session):
        """Calcular estadísticas generales de estudiantes"""
        # Total de estudiantes
        total_estudiantes = db.query(Usuario).count()
        
        # Promedio general de puntajes
        avg_result = db.query(func.avg(
            case(
                (Puntaje.puntaje_total > 0, (Puntaje.puntaje_obtenido * 100.0) / Puntaje.puntaje_total),
                else_=0
            )
        )).scalar()
        promedio_general = round(float(avg_result or 0), 2)
        
        # Estudiantes activos (con puntaje > 0)
        estudiantes_activos = db.query(Puntaje).filter(Puntaje.puntaje_obtenido > 0).count()
        
        # Distribución por niveles
        nivel_counts = db.query(
            Puntaje.nivel, 
            func.count(Puntaje.nivel)
        ).group_by(Puntaje.nivel).all()
        
        distribucion_niveles = {nivel: count for nivel, count in nivel_counts}
        
        return {
            "promedio_general": promedio_general,
            "total_estudiantes": total_estudiantes,
            "estudiantes_activos": estudiantes_activos,
            "distribucion_niveles": distribucion_niveles
        }

    def get_students_progress_data(db: Session):
        """Obtener datos de progreso de todos los estudiantes"""
        # Query con JOIN para obtener datos completos
        results = db.query(
            Usuario.username,
            Usuario.email,
            Puntaje.puntaje_obtenido,
            Puntaje.puntaje_total,
            Puntaje.nivel,
            func.count(Visita.id).label('lecciones_completadas')
        ).outerjoin(
            Puntaje, Usuario.id == Puntaje.usuario_id
        ).outerjoin(
            Visita, Usuario.id == Visita.usuario_id
        ).group_by(
            Usuario.id, Usuario.username, Usuario.email,
            Puntaje.puntaje_obtenido, Puntaje.puntaje_total, Puntaje.nivel
        ).all()
        
        estudiantes = []
        for result in results:
            # Simular fecha de última actividad (puedes agregar campo real si necesitas)
            fecha_actividad = datetime.now().strftime("%Y-%m-%d")
            
            estudiante = {
                "username": result.username,
                "email": result.email,
                "puntaje_obtenido": result.puntaje_obtenido or 0,
                "puntaje_total": result.puntaje_total or 20,
                "nivel": result.nivel or "Básico",
                "fecha_ultima_actividad": fecha_actividad,
                "lecciones_completadas": result.lecciones_completadas or 0
            }
            estudiantes.append(estudiante)
        
        return {
            "estudiantes": estudiantes,
            "total": len(estudiantes)
        }    

    # Funciones auxiliares para docentes
    def get_teacher_by_email(db: Session, email: str):
        return db.query(Docente).filter(Docente.email == email).first()

    def get_teacher_by_username(db: Session, username: str):
        return db.query(Docente).filter(Docente.username == username).first()

    def create_teacher(db: Session, username: str, email: str, password: str):
        db_teacher = Docente(username=username, email=email, password=password, institucion="CCJCALLAO")
        db.add(db_teacher)
        db.commit()
        db.refresh(db_teacher)
        return db_teacher


    # Endpoints de usuarios
    @app.get("/usuarios", response_model=List[UsuarioResponse])
    async def get_usuarios(db: Session = Depends(get_db)):
        """Obtener todos los usuarios con username incluido"""
        usuarios = db.query(Usuario).all()
        return [{"username": user.username, "email": user.email, "password": user.password} for user in usuarios]

    @app.post("/usuarios/registro", response_model=RegistroResponse)
    async def registrar_usuario(usuario: UsuarioRegistro, db: Session = Depends(get_db)):
        """Registrar un nuevo usuario"""
        if get_user_by_email(db, usuario.email):
            raise HTTPException(status_code=400, detail="El email ya está registrado")
        if get_user_by_username(db, usuario.username):
            raise HTTPException(status_code=400, detail="El username ya está registrado")
        nuevo_usuario = create_user(db, usuario.username, usuario.email, usuario.password)
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
        return {"username": usuario.username, "email": usuario.email}

    @app.delete("/usuarios/{email}", response_model=MessageResponse)
    async def eliminar_usuario(email: str, db: Session = Depends(get_db)):
        """Eliminar un usuario y todos sus datos relacionados"""
        usuario = get_user_by_email(db, email)
        if not usuario:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        db.delete(usuario)
        db.commit()
        return {"message": "Usuario, favoritos, visitas y puntajes eliminados exitosamente"}

    @app.put("/usuarios/{email}/perfil", response_model=MessageResponse)
    async def actualizar_perfil_usuario(email: str, perfil_actualizado: UsuarioUpdateProfile, db: Session = Depends(get_db)):
        """Actualizar perfil de usuario (solo username)"""
        usuario = get_user_by_email(db, email)
        if not usuario:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        if perfil_actualizado.username != usuario.username:
            usuario_existente = get_user_by_username(db, perfil_actualizado.username)
            if usuario_existente:
                raise HTTPException(status_code=400, detail="El nuevo username ya está registrado")
        usuario.username = perfil_actualizado.username
        db.commit()
        return {"message": "Perfil actualizado exitosamente"}

    @app.put("/usuarios/{email}/password", response_model=MessageResponse)
    async def cambiar_contraseña_usuario(email: str, password_data: UsuarioChangePassword, db: Session = Depends(get_db)):
        """Cambiar contraseña de usuario"""
        usuario = get_user_by_email(db, email)
        if not usuario:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        usuario.password = password_data.new_password
        db.commit()
        return {"message": "Contraseña actualizada correctamente"}



    # Endpoints de favoritos
    @app.post("/usuarios/{email}/favoritos", response_model=FavoritoAddResponse)
    async def agregar_favorito(email: str, favorito: FavoritoRequest, db: Session = Depends(get_db)):
        """Agregar una clase a favoritos"""
        usuario = get_user_by_email(db, email)
        if not usuario:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        favorito_existente = db.query(Favorito).filter(
            Favorito.usuario_id == usuario.id,
            Favorito.clase_id == favorito.clase_id
        ).first()
        if favorito_existente:
            raise HTTPException(status_code=400, detail="La clase ya está en favoritos")
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
        
        visita_existente = db.query(Visita).filter(
            Visita.usuario_id == usuario.id,
            Visita.clase_id == visita.clase_id
        ).first()
        
        if visita_existente:
            visita_existente.count += 1
            db.commit()
        else:
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
        nivel_usado = puntaje_request.nivel
        
        try:
            ml_prediction = predictor.predict_level(
                score_obtained=puntaje_request.puntaje_obtenido,
                total_score=puntaje_request.puntaje_total
            )
            nivel_ml = ml_prediction['nivel_predicho']
            confianza = ml_prediction.get('confianza', 0)
            if confianza > 0.7 or _compare_levels(nivel_ml, puntaje_request.nivel):
                nivel_usado = nivel_ml
        except Exception as e:
            print(f"Error en predicción ML, usando nivel manual: {e}")
        
        if not puntaje:
            puntaje = Puntaje(
                usuario_id=usuario.id,
                puntaje_obtenido=puntaje_request.puntaje_obtenido,
                puntaje_total=puntaje_request.puntaje_total,
                nivel=nivel_usado
            )
            db.add(puntaje)
            is_new_best = True
        else:
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
            return {
                "modelo_cargado": True,
                "estadisticas": stats
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error obteniendo estadísticas: {str(e)}")


    @app.get("/usuarios/{email}/puntajes/modelo/predict", response_model=ModelPredictUserResponse)
    async def predict_user_english_level(email: str, db: Session = Depends(get_db)):
        """Predecir nivel de inglés para un usuario específico usando su puntaje guardado"""
        usuario = get_user_by_email(db, email)
        if not usuario:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        puntaje = db.query(Puntaje).filter(Puntaje.usuario_id == usuario.id).first()
        if not puntaje:
            raise HTTPException(status_code=404, detail="No se encontraron puntajes para este usuario")
        
        try:
            prediction_result = predictor.predict_level(
                score_obtained=puntaje.puntaje_obtenido,
                total_score=puntaje.puntaje_total
            )
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
        usuario = get_user_by_email(db, email)
        if not usuario:
            raise HTTPException(status_code=404, detail="Usuario no encontrado")
        
        puntaje = db.query(Puntaje).filter(Puntaje.usuario_id == usuario.id).first()
        if not puntaje:
            raise HTTPException(status_code=404, detail="No se encontraron puntajes para este usuario")
        
        try:
            prediction_result = predictor.predict_level(
                score_obtained=puntaje.puntaje_obtenido,
                total_score=puntaje.puntaje_total
            )
            model_stats = predictor.get_model_stats()
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



    # Endpoints para docentes
    @app.get("/docentes", response_model=List[DocenteResponse])
    async def get_docentes(db: Session = Depends(get_db)):
        """Obtener todos los docentes"""
        docentes = db.query(Docente).all()
        return [{"username": d.username, "email": d.email, "password": d.password, "institucion": d.institucion} for d in docentes]

    @app.post("/docentes/registro", response_model=RegistroResponse)
    async def registrar_docente(docente: DocenteRegistro, db: Session = Depends(get_db)):
        """Registrar un nuevo docente"""
        # Verificar si el email ya existe
        if get_teacher_by_email(db, docente.email):
            raise HTTPException(status_code=400, detail="El email ya está registrado")
        
        # Verificar si el username ya existe
        if get_teacher_by_username(db, docente.username):
            raise HTTPException(status_code=400, detail="El username ya está registrado")
        
        # Validar dominio de email institucional
        if not docente.email.endswith("@ccjcallao.edu.pe"):
            raise HTTPException(status_code=400, detail="Debe usar un correo institucional @ccjcallao.edu.pe")
        
        # Crear nuevo docente
        nuevo_docente = create_teacher(db, docente.username, docente.email, docente.password)
        
        return {"message": "Docente registrado exitosamente", "email": docente.email}

    @app.post("/docentes/login", response_model=LoginResponse)
    async def login_docente(docente: DocenteLogin, db: Session = Depends(get_db)):
        """Autenticar docente"""
        docente_encontrado = get_teacher_by_email(db, docente.email)
        
        if not docente_encontrado:
            raise HTTPException(status_code=404, detail="Docente no encontrado")
        
        if docente_encontrado.password != docente.password:
            raise HTTPException(status_code=401, detail="Contraseña incorrecta")
        
        return {"message": "Login exitoso", "email": docente.email}

    @app.get("/docentes/{email}", response_model=DocenteInfo)
    async def obtener_docente(email: str, db: Session = Depends(get_db)):
        """Obtener información de un docente específico"""
        docente = get_teacher_by_email(db, email)
        
        if not docente:
            raise HTTPException(status_code=404, detail="Docente no encontrado")
        
        return {"username": docente.username, "email": docente.email, "institucion": docente.institucion}


    @app.get("/docentes/estadisticas/estudiantes")
    async def get_students_statistics(db: Session = Depends(get_db)):
        """Obtener estadísticas generales de estudiantes para el dashboard docente"""
        try:
            stats = calculate_general_stats(db)
            return stats
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al obtener estadísticas: {str(e)}")

    @app.get("/docentes/progreso/estudiantes")
    async def get_students_progress(db: Session = Depends(get_db)):
        """Obtener progreso detallado de todos los estudiantes"""
        try:
            progress_data = get_students_progress_data(db)
            return progress_data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al obtener progreso: {str(e)}")

    @app.get("/docentes/dashboard/completo")
    async def get_teacher_dashboard_complete(db: Session = Depends(get_db)):
        """Obtener datos completos del dashboard docente en una sola llamada"""
        try:
            # Obtener estadísticas generales
            estadisticas_generales = calculate_general_stats(db)
            
            # Obtener progreso de estudiantes
            progreso_estudiantes = get_students_progress_data(db)
            
            return {
                "estadisticas_generales": estadisticas_generales,
                "progreso_estudiantes": progreso_estudiantes
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al obtener dashboard: {str(e)}")

    @app.get("/docentes/estudiantes/nivel/{nivel}")
    async def get_students_by_level(nivel: str, db: Session = Depends(get_db)):
        """Obtener estudiantes filtrados por nivel específico"""
        try:
            # Validar nivel
            niveles_validos = ["Básico", "Intermedio", "Avanzado"]
            if nivel not in niveles_validos:
                raise HTTPException(status_code=400, detail="Nivel no válido")
            
            # Query para obtener estudiantes del nivel específico
            results = db.query(
                Usuario.username,
                Usuario.email,
                Puntaje.puntaje_obtenido,
                Puntaje.puntaje_total,
                Puntaje.nivel,
                func.count(Visita.id).label('lecciones_completadas')
            ).join(
                Puntaje, Usuario.id == Puntaje.usuario_id
            ).outerjoin(
                Visita, Usuario.id == Visita.usuario_id
            ).filter(
                Puntaje.nivel == nivel
            ).group_by(
                Usuario.id, Usuario.username, Usuario.email,
                Puntaje.puntaje_obtenido, Puntaje.puntaje_total, Puntaje.nivel
            ).all()
            
            estudiantes = []
            for result in results:
                estudiante = {
                    "username": result.username,
                    "email": result.email,
                    "puntaje_obtenido": result.puntaje_obtenido,
                    "puntaje_total": result.puntaje_total,
                    "nivel": result.nivel,
                    "fecha_ultima_actividad": datetime.now().strftime("%Y-%m-%d"),
                    "lecciones_completadas": result.lecciones_completadas
                }
                estudiantes.append(estudiante)
            
            return {
                "nivel": nivel,
                "estudiantes": estudiantes,
                "total": len(estudiantes)
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al filtrar estudiantes: {str(e)}")

    @app.get("/docentes/estudiantes/top/{limit}")
    async def get_top_students(limit: int = 5, db: Session = Depends(get_db)):
        """Obtener los mejores estudiantes ordenados por porcentaje"""
        try:
            if limit > 20:
                limit = 20  # Limitar máximo
            
            # Query para obtener top estudiantes ordenados por porcentaje
            results = db.query(
                Usuario.username,
                Usuario.email,
                Puntaje.puntaje_obtenido,
                Puntaje.puntaje_total,
                Puntaje.nivel,
                func.count(Visita.id).label('lecciones_completadas'),
                case(
                    (Puntaje.puntaje_total > 0, (Puntaje.puntaje_obtenido * 100.0) / Puntaje.puntaje_total),
                    else_=0
                ).label('porcentaje')
            ).join(
                Puntaje, Usuario.id == Puntaje.usuario_id
            ).outerjoin(
                Visita, Usuario.id == Visita.usuario_id
            ).group_by(
                Usuario.id, Usuario.username, Usuario.email,
                Puntaje.puntaje_obtenido, Puntaje.puntaje_total, Puntaje.nivel
            ).order_by(
                desc('porcentaje')
            ).limit(limit).all()
            
            estudiantes = []
            for i, result in enumerate(results, 1):
                estudiante = {
                    "posicion": i,
                    "username": result.username,
                    "email": result.email,
                    "puntaje_obtenido": result.puntaje_obtenido,
                    "puntaje_total": result.puntaje_total,
                    "nivel": result.nivel,
                    "porcentaje": round(float(result.porcentaje), 2),
                    "fecha_ultima_actividad": datetime.now().strftime("%Y-%m-%d"),
                    "lecciones_completadas": result.lecciones_completadas
                }
                estudiantes.append(estudiante)
            
            return {
                "top_estudiantes": estudiantes,
                "limite": limit,
                "total": len(estudiantes)
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al obtener top estudiantes: {str(e)}")
        

    @app.get("/docentes/estadisticas/detalladas")
    async def get_detailed_statistics(db: Session = Depends(get_db)):
        """Obtener estadísticas detalladas para el dashboard docente"""
        try:
            # Estadísticas básicas
            total_estudiantes = db.query(Usuario).count()
            total_docentes = db.query(Docente).count()
            
            # Estadísticas de puntajes
            puntajes_stats = db.query(
                func.avg(Puntaje.puntaje_obtenido).label('promedio_puntaje'),
                func.max(Puntaje.puntaje_obtenido).label('max_puntaje'),
                func.min(Puntaje.puntaje_obtenido).label('min_puntaje'),
                func.count(Puntaje.id).label('total_evaluaciones')
            ).first()
            
            # Distribución por niveles con porcentajes
            nivel_counts = db.query(
                Puntaje.nivel, 
                func.count(Puntaje.nivel).label('cantidad')
            ).group_by(Puntaje.nivel).all()
            
            distribucion_niveles = {}
            for nivel, cantidad in nivel_counts:
                porcentaje = (cantidad / total_estudiantes * 100) if total_estudiantes > 0 else 0
                distribucion_niveles[nivel] = {
                    "cantidad": cantidad,
                    "porcentaje": round(porcentaje, 2)
                }
            
            # Actividad reciente (estudiantes con puntaje en últimos 30 días)
            # Como no tienes fecha en puntajes, simulamos actividad
            estudiantes_activos_recientes = db.query(Puntaje).filter(
                Puntaje.puntaje_obtenido > 0
            ).count()
            
            return {
                "resumen": {
                    "total_estudiantes": total_estudiantes,
                    "total_docentes": total_docentes,
                    "total_evaluaciones": puntajes_stats.total_evaluaciones or 0,
                    "estudiantes_activos": estudiantes_activos_recientes
                },
                "puntajes": {
                    "promedio": round(float(puntajes_stats.promedio_puntaje or 0), 2),
                    "maximo": puntajes_stats.max_puntaje or 0,
                    "minimo": puntajes_stats.min_puntaje or 0
                },
                "distribucion_niveles": distribucion_niveles,
                "fecha_consulta": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al obtener estadísticas detalladas: {str(e)}")



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
