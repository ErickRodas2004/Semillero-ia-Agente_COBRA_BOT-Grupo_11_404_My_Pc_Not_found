# IMPORTANTE 
# No olvide importar Su API en el main "agente_cobranza"
import sys
import os

import pandas as pd
import time
from langchain_core.messages import HumanMessage

# --- 🛠️ CONFIGURACIÓN DE RUTAS (Adaptada a tu estructura src/tests) ---

ruta_actual_tests = os.path.dirname(os.path.abspath(__file__))

ruta_src = os.path.dirname(ruta_actual_tests)

sys.path.append(ruta_src)

print(f"📍 Ejecutando desde: {ruta_actual_tests}")
print(f"🔍 Buscando 'agente_cobranza.py' en: {ruta_src}")

try:
    from agente_cobranza import (
        registrar_cliente, 
        leer_base_datos, 
        actualizar_deuda, 
        eliminar_cliente_pagado,
        app, 
        FILE_PATH
    )
    print("✅ Archivo 'agente_cobranza.py' importado correctamente.\n")

except ImportError as e:
    print(f"\n❌ ERROR: {e}")
    print("Verifica que el archivo 'agente_cobranza.py' esté justo afuera de la carpeta 'tests'.")
    sys.exit(1)


def limpiar_entorno():
    """Elimina el archivo CSV para empezar las pruebas desde cero."""
    if os.path.exists(FILE_PATH):
        os.remove(FILE_PATH)
        print("🗑️ Base de datos eliminada para iniciar pruebas limpias.")

def test_herramientas_directas():
    print("\n--- 🛠️ INICIANDO TEST DE HERRAMIENTAS (BACKEND) ---")
    
    # 1. Prueba de Registro
    print(f"👉 Probando registro manual...")
    res = registrar_cliente.invoke({
        "nombre": "Test User", 
        "deuda": 500.0, 
        "dias_mora": 30, 
        "producto": "Tarjeta Crédito"
    })
    print(f"Resultado: {res}")
    assert "CLIENTE_REGISTRADO" in res, "❌ Falló el registro"

    # 2. Prueba de Lectura
    print(f"👉 Probando lectura de DB...")
    res_lectura = leer_base_datos.invoke({})
    print(f"Resultado: {res_lectura}")
    assert "TABLA_DATOS" in res_lectura, "❌ Falló la lectura"

    # 3. Prueba de Actualización
    print(f"👉 Probando actualización de deuda...")
    res_update = actualizar_deuda.invoke({
        "nombre": "Test User", 
        "nueva_deuda": 200.0
    })
    print(f"Resultado: {res_update}")
    assert "DATOS_ACTUALIZADOS" in res_update, "❌ Falló la actualización"

    print("✅ TODAS LAS HERRAMIENTAS FUNCIONAN CORRECTAMENTE.")

def test_agente_inteligente():
    print("\n--- 🧠 INICIANDO TEST DEL AGENTE (LANGGRAPH) ---")
    
    # Simulamos una entrada de usuario sin usar Flet
    input_text = "Registra a María López con una deuda de 1200 dólares por un Préstamo Personal y tiene 60 días de mora."
    
    print(f"👤 Usuario dice: '{input_text}'")
    print("⏳ Procesando con el Agente (esto puede tardar unos segundos)...")
    
    config = {"configurable": {"thread_id": "test_script_1"}}
    inputs = {"messages": [HumanMessage(content=input_text)]}
    
    # Ejecutamos el grafo (el cerebro del bot)
    output = app.invoke(inputs, config)
    
    # Obtenemos la última respuesta del bot
    bot_response = output["messages"][-1].content
    
    print("\n🤖 Respuesta del Bot:")
    print("-" * 50)
    print(bot_response)
    print("-" * 50)
    
    # Verificaciones básicas del speech generado
    if "María López" in bot_response and "1200" in bot_response:
        print("✅ El speech contiene los datos correctos.")
    else:
        print("⚠️ Advertencia: El speech podría no tener los datos personalizados.")

if __name__ == "__main__":
    try:
        limpiar_entorno()
        test_herramientas_directas()
        test_agente_inteligente()
        print("\n🚀 FIN DE LAS PRUEBAS: El sistema parece estable.")
    except ImportError:
        print("❌ ERROR: No se encontró 'main.py'. Asegúrate de guardar tu código original con ese nombre.")
    except Exception as e:
        print(f"❌ ERROR FATAL DURANTE LAS PRUEBAS: {e}")