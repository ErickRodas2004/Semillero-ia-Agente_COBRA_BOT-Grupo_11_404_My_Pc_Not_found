#pip install flet pandas langchain-google-genai langgraph
import os
import pandas as pd
import asyncio
from datetime import datetime
from typing import Annotated, TypedDict, List

import flet as ft

# LangChain & LangGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# ⚙️ CONFIGURACIÓN
# ==========================================
API_KEY = os.environ.get("GOOGLE_API_KEY", "-------")
os.environ["GOOGLE_API_KEY"] = API_KEY

FILE_PATH = "clientes.csv"

# ==========================================
# 🛠️ HERRAMIENTAS
# ==========================================

@tool
def registrar_cliente(nombre: str, deuda: float, dias_mora: int, producto: str):
    """
    Registra un nuevo cliente deudor en la base de datos CSV.
    """
    if not os.path.exists(FILE_PATH):
        pd.DataFrame(columns=[
            'nombre', 'deuda', 'dias_mora', 'producto', 'fecha_registro'
        ]).to_csv(FILE_PATH, index=False)

    df = pd.read_csv(FILE_PATH)
    df.loc[len(df)] = [
        nombre, deuda, dias_mora, producto,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ]
    df.to_csv(FILE_PATH, index=False)
    return f"CLIENTE_REGISTRADO | nombre={nombre} | deuda={deuda}"

@tool
def eliminar_cliente_pagado(nombre: str):
    """
    Elimina un cliente cuando ya pagó.
    """
    if not os.path.exists(FILE_PATH):
        return "BD_NO_EXISTE"

    df = pd.read_csv(FILE_PATH)
    mask = df['nombre'].str.lower().str.strip() == nombre.lower().strip()

    if not mask.any():
        return "CLIENTE_NO_ENCONTRADO"

    df = df[~mask]
    df.to_csv(FILE_PATH, index=False)
    
    # Retornar solo confirmación
    return f"CLIENTE_ELIMINADO | nombre={nombre}"

@tool
def leer_base_datos():
    """
    Lista clientes. Retorna TABLA_DATOS seguido de la tabla.
    """
    if not os.path.exists(FILE_PATH):
        return "BD_VACIA"

    df = pd.read_csv(FILE_PATH)
    if df.empty:
        return "SIN_CLIENTES"

    return f"TABLA_DATOS\n{df.to_markdown(index=False)}"

@tool
def actualizar_deuda(nombre: str, nueva_deuda: float = None, nuevos_dias_mora: int = None):
    """
    Actualiza el monto de deuda y/o días de mora de un cliente existente.
    Se pueden actualizar ambos campos o solo uno de ellos.
    """
    if not os.path.exists(FILE_PATH):
        return "BD_NO_EXISTE"

    df = pd.read_csv(FILE_PATH)
    mask = df['nombre'].str.lower().str.strip() == nombre.lower().strip()

    if not mask.any():
        return "CLIENTE_NO_ENCONTRADO"

    # Actualizar campos según lo que se proporcione
    cambios = []
    if nueva_deuda is not None:
        df.loc[mask, 'deuda'] = nueva_deuda
        cambios.append(f"deuda={nueva_deuda}")
    
    if nuevos_dias_mora is not None:
        df.loc[mask, 'dias_mora'] = nuevos_dias_mora
        cambios.append(f"dias_mora={nuevos_dias_mora}")
    
    if not cambios:
        return "NO_SE_ESPECIFICARON_CAMBIOS"
    
    df.to_csv(FILE_PATH, index=False)
    
    cambios_str = " | ".join(cambios)
    return f"DATOS_ACTUALIZADOS | nombre={nombre} | {cambios_str}"

# ==========================================
# 🧠 AGENTE
# ==========================================

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",   
    temperature=0.7,            
    max_output_tokens=700
)

llm_with_tools = llm.bind_tools([
    registrar_cliente,
    eliminar_cliente_pagado,
    leer_base_datos,
    actualizar_deuda
])

def agent_node(state: AgentState):
    system_prompt = """
ERES COBRA-BOT AI, un EXPERTO en generación de Speech de cobranza hiperpersonalizados y persuasivos.

TU FUNCIÓN PRINCIPAL:
Generar mensajes de cobranza creativos, hiperpersonalizados y altamente persuasivos basados en los datos de cada cliente.
Cada speech debe ser único y adaptado al perfil del deudor.

GESTIÓN DE DATOS (Función secundaria):
Puedes registrar, consultar, actualizar y eliminar clientes cuando el usuario lo solicite explícitamente.

REGLA CRÍTICA - MOSTRAR TABLAS:
- La tabla SOLO se muestra cuando el usuario EXPLÍCITAMENTE pide verla
- Palabras clave: "consultar", "mostrar", "ver", "listar", "muéstrame", "dame los registros"
- Para registros, actualizaciones o eliminaciones → NO mostrar tabla automáticamente
- Solo confirma la acción sin mostrar la tabla

GENERACIÓN DE SPEECH - REGLAS OBLIGATORIAS:

1. CUANDO GENERAS UN SPEECH DE COBRANZA:

   A) PERSONALIZACIÓN TOTAL:
   - Usa SIEMPRE el nombre del cliente
   - Menciona el producto específico
   - Incluye el monto exacto de la deuda
   - Usa espacios para poner manualmente información de contacto, nombre o empresa que pide el speech (no pongas información falsa)
   - Referencia los días de mora
   - Se creativo con cada speech
   - NO seas redundante

   B) TONO ADAPTATIVO según deuda y mora:
   * Deuda baja <= $300 y mora <= 15 días : Empático, cordial, recordatorio amable
   * Deuda media $300-$1000 o mora 16-44 días : Firme, profesional, urgente pero respetuoso
   * Deuda alta >= $1001 o mora >= 45 días : Serio, directo, menciona consecuencias legales

   C) ESTRUCTURA DEL SPEECH (mínimo 3 párrafos máximo 4):
   - Párrafo 1: Saludo personalizado + identificación de la deuda
   - Párrafo 2: Urgencia + beneficios de pagar ahora + facilidades
   - Párrafo 3: Llamado a la acción claro + datos de contacto/pago
   - Parrrafo 4(de ser necesario): mensaje de ánimos si la deuda es alta
   

   D) TÉCNICAS DE PERSUASIÓN:
   - Usa gatillos emocionales (responsabilidad, tranquilidad, beneficios)
   - Menciona consecuencias de no pagar (sin amenazar)
   - Ofrece soluciones (planes de pago, descuentos)
   - Crea urgencia (plazos, recargos)
   - Lenguaje positivo y profesional


2. CUANDO REGISTRAS UN CLIENTE:
   - Confirma el registro
   - Genera automáticamente un speech de cobranza personalizado para ese cliente
   - El speech debe seguir TODAS las reglas anteriores

3. CUANDO CONSULTAS LA BASE DE DATOS:
   - Solo usa la herramienta cuando el usuario pida EXPLÍCITAMENTE ver/consultar/mostrar
   - NO incluyas tablas markdown en tu respuesta
   - Di algo breve como: "Aquí están tus registros actuales."
   - La tabla se mostrará automáticamente en la interfaz

4. CUANDO REGISTRAS, ACTUALIZAS O ELIMINAS:
   - Confirma la acción brevemente
   - NO llames a leer_base_datos automáticamente
   - NO muestres tabla a menos que el usuario la pida
   - Ejemplo: "Cliente registrado. ¿Quieres ver tus registros actualizados?"

5. PARA SALUDOS O CONVERSACIÓN CASUAL:
   - Responde amigablemente
   - NO uses NINGUNA herramienta
   - Simplemente saluda y pregunta en qué puedes ayudar
   - Ejemplos: "hola", "buenos días", "hey", "qué tal"

PROHIBIDO:
- Incluir tablas markdown (|, ---) en TUS respuestas de texto
- Mencionar "aquí está la tabla" o referencias similares
- Consultar base de datos automáticamente después de registrar/actualizar/eliminar
- Usar herramientas sin que el usuario lo pida EXPLÍCITAMENTE
- Generar speech genéricos o repetitivos

RECUERDA: 
- La tabla solo se muestra cuando el usuario PIDE verla explícitamente
- Después de registrar/actualizar/eliminar → Solo confirma, NO muestres tabla
- El usuario decidirá cuándo quiere ver sus registros
- Tu trabajo es generar speech personalizados y confirmar acciones
"""

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}

def tools_node(state: AgentState):
    last = state["messages"][-1]
    responses = []

    for call in last.tool_calls:
        name, args = call["name"], call["args"]

        if name == "registrar_cliente":
            res = registrar_cliente.invoke(args)
        elif name == "eliminar_cliente_pagado":
            res = eliminar_cliente_pagado.invoke(args)
        elif name == "leer_base_datos":
            res = leer_base_datos.invoke(args)
        elif name == "actualizar_deuda":
            res = actualizar_deuda.invoke(args)
        else:
            res = "ERROR_TOOL"

        responses.append(
            ToolMessage(
                content=res,
                tool_call_id=call["id"]
            )
        )

    return {"messages": responses}

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tools_node)
workflow.set_entry_point("agent")

def route(state: AgentState):
    return "tools" if state["messages"][-1].tool_calls else END

workflow.add_conditional_edges("agent", route)
workflow.add_edge("tools", "agent")

app = workflow.compile(checkpointer=MemorySaver())

# ==========================================
# 🎨 UI CON FLET
# ==========================================

def main(page: ft.Page):
    page.title = "COBRA-BOT AI - Speech Generator"
    page.window_width = 500
    page.window_height = 800
    page.window_always_on_top = True
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 0
    page.bgcolor = "#0A0E27"

    chat = ft.ListView(
        expand=True,
        spacing=10,
        auto_scroll=True,
        scroll=ft.ScrollMode.ADAPTIVE,
        padding=10
    )

    input_box = ft.TextField(
        hint_text="Escribe tu solicitud o pide generar un speech...",
        expand=True,
        on_submit=lambda e: enviar(),
        border_color="#475569",
        focused_border_color="#3B82F6",
        bgcolor="#0F172A",
        text_style=ft.TextStyle(color="#F1F5F9"),
        hint_style=ft.TextStyle(color="#64748B"),
        content_padding=15,
        multiline=False,
        max_lines=1
    )

    thread_counter = {"count": 0}

    def limpiar_chat():
        """Limpia el historial del chat y reinicia la conversación"""
        chat.controls.clear()
        thread_counter["count"] += 1
        page.update()
        mensaje("✨ Chat reiniciado. ¿En qué puedo ayudarte ahora?", "bot")

    def render_table(md):
        """Renderiza una tabla markdown como DataTable de Flet."""
        try:
            lines = [line for line in md.split("\n") if line.strip()]
            
            if len(lines) < 2:
                return ft.Markdown(md, selectable=True)
            
            headers = [h.strip() for h in lines[0].split("|")[1:-1]]
            
            data_rows = []
            for line in lines[2:]:
                if "|" in line:
                    cells = [c.strip() for c in line.split("|")[1:-1]]
                    if len(cells) == len(headers):
                        data_rows.append(cells)
            
            if not data_rows:
                return ft.Markdown(md, selectable=True)
            
            return ft.DataTable(
                columns=[ft.DataColumn(ft.Text(h, weight="bold", color="#F1F5F9")) for h in headers],
                rows=[
                    ft.DataRow(cells=[ft.DataCell(ft.Text(c, color="#E2E8F0")) for c in row])
                    for row in data_rows
                ],
                border=ft.border.all(1, "#475569"),
                border_radius=8,
                heading_row_color="#0F172A",
                data_row_color={"hovered": "#1E293B"},
            )
        except Exception as e:
            print(f"⚠️ Error rendering table: {e}")
            return ft.Markdown(md, selectable=True, extension_set=ft.MarkdownExtensionSet.GITHUB_WEB)

    def mensaje(texto, autor="bot"):
        """Muestra un mensaje en el chat. Solo renderiza tabla si tiene el marcador TABLA_DATOS"""
        
        # Si el texto contiene TABLA_DATOS, extraer solo la tabla
        if isinstance(texto, str) and "TABLA_DATOS" in texto:
            partes = texto.split("TABLA_DATOS")
            if len(partes) > 1:
                tabla_md = partes[1].strip()
                bubble_content = render_table(tabla_md)
            else:
                return  # No mostrar nada si no hay tabla después del marcador
        else:
            # Texto normal sin tabla
            bubble_content = ft.Markdown(
                texto,
                selectable=True,
                extension_set=ft.MarkdownExtensionSet.GITHUB_WEB
            )
        
        chat.controls.append(
            ft.Row(
                alignment=ft.MainAxisAlignment.END if autor == "user" else ft.MainAxisAlignment.START,
                controls=[
                    ft.Container(
                        width=page.width * 0.85,
                        content=bubble_content,
                        bgcolor="#1E293B" if autor == "bot" else "#3B82F6",
                        padding=15,
                        border_radius=15,
                        shadow=ft.BoxShadow(
                            spread_radius=1,
                            blur_radius=10,
                            color=ft.Colors.with_opacity(0.2, "#000000"),
                            offset=ft.Offset(0, 2),
                        )
                    )
                ]
            )
        )
        page.update()
        
    async def procesar(texto):
        loading = None
        try:
            loading = ft.Text("⏳ Generando respuesta...", italic=True, color="#94A3B8")
            chat.controls.append(loading)
            page.update()

            inputs = {"messages": [HumanMessage(content=texto)]}
            config = {"configurable": {"thread_id": f"ui_flet_{thread_counter['count']}"}}

            result = await asyncio.to_thread(app.invoke, inputs, config)

            if loading in chat.controls:
                chat.controls.remove(loading)
                page.update()

            # CONTROL TOTAL: Solo mostrar tabla si el usuario EXPLÍCITAMENTE pidió consultar
            palabras_clave_consulta = [
                "consultar", "mostrar", "ver", "listar", "muestra", "dame",
                "cuál", "cuáles", "qué clientes", "mis registros", "base de datos",
                "tabla", "clientes"
            ]
            
            solicita_tabla = any(palabra in texto.lower() for palabra in palabras_clave_consulta)
            
            # Solo procesar ToolMessages con tablas si el usuario las solicitó
            if solicita_tabla:
                tool_messages_con_tabla = []
                for msg in reversed(result["messages"]):
                    if isinstance(msg, ToolMessage) and "TABLA_DATOS" in msg.content:
                        tool_messages_con_tabla.append(msg)
                
                # Mostrar solo la tabla más reciente
                if tool_messages_con_tabla:
                    mensaje(tool_messages_con_tabla[0].content, "bot")

            # Obtener respuesta final del bot
            texto_final = result["messages"][-1].content if result["messages"][-1].content else ""

            # Mostrar respuesta del bot (sin tablas, ya que no debe incluirlas)
            if texto_final and texto_final.strip():
                # Limpiar cualquier tabla que pudiera haber en el texto final
                if "|" not in texto_final or "---" not in texto_final:
                    # No tiene tabla, mostrar normal
                    mensaje(texto_final, "bot")
                else:
                    # Si por alguna razón tiene tabla, extraer solo texto
                    lineas = texto_final.split("\n")
                    texto_sin_tabla = []
                    for linea in lineas:
                        if "|" not in linea and "---" not in linea:
                            texto_sin_tabla.append(linea)
                    
                    texto_limpio = "\n".join(texto_sin_tabla).strip()
                    if texto_limpio:
                        mensaje(texto_limpio, "bot")

            input_box.disabled = False
            page.update()

        except Exception as e:
            if loading and loading in chat.controls:
                chat.controls.remove(loading)
                page.update()
            mensaje(f" Error: {e}", "bot")
            input_box.disabled = False
            page.update()

    def enviar():
        texto = input_box.value.strip()
        if not texto:
            return

        input_box.value = ""
        input_box.disabled = True
        mensaje(texto, "user")

        page.run_task(procesar, texto)

    page.add(
        ft.Container(
            expand=True,
            bgcolor="#0A0E27",
            content=ft.Column(
                expand=True,
                spacing=0,
                controls=[
                    # Header
                    ft.Container(
                        bgcolor="#1E293B",
                        padding=15,
                        content=ft.Row(
                            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                            controls=[
                                ft.Row(
                                    spacing=10,
                                    controls=[
                                        ft.Icon(ft.Icons.CHAT_BUBBLE, color="#3B82F6", size=24),
                                        ft.Text(
                                            "COBRA-BOT AI",
                                            size=20,
                                            weight="bold",
                                            color="#F1F5F9"
                                        ),
                                    ]
                                ),
                                ft.IconButton(
                                    icon=ft.Icons.DELETE_SWEEP,
                                    tooltip="Limpiar chat",
                                    icon_color="#94A3B8",
                                    on_click=lambda e: limpiar_chat()
                                )
                            ]
                        )
                    ),
                    # Subtitle
                    ft.Container(
                        bgcolor="#0F172A",
                        padding=ft.padding.only(left=15, right=15, top=8, bottom=8),
                        content=ft.Text(
                            "🎯 Generador de Speech Hiperpersonalizados",
                            size=12,
                            color="#94A3B8",
                            italic=True
                        )
                    ),
                    # Chat área
                    ft.Container(
                        expand=True,
                        bgcolor="#0A0E27",
                        content=chat
                    ),
                    # Input área
                    ft.Container(
                        bgcolor="#1E293B",
                        padding=12,
                        content=ft.Row(
                            spacing=10,
                            controls=[
                                input_box,
                                ft.Container(
                                    content=ft.TextButton(
                                        content=ft.Row(
                                            spacing=5,
                                            controls=[
                                                ft.Icon(ft.Icons.SEND, size=18),
                                                ft.Text("Enviar", size=14)
                                            ]
                                        ),
                                        style=ft.ButtonStyle(
                                            bgcolor="#3B82F6",
                                            color="white",
                                            padding=15,
                                        ),
                                        on_click=lambda e: enviar()
                                    )
                                )
                            ]
                        )
                    )
                ]
            )
        )
    )

    # --- MODIFICACIÓN: SALUDO INICIAL ---
    # Mostramos el mensaje de bienvenida al cargar la UI
    mensaje(
        """👋 **¡Hola! Soy COBRA-BOT AI**

Soy tu asistente inteligente especializado en gestión de cobranzas.

Mis principales características son:

🔹 **Generación de Speeches:** Creo guiones de cobro persuasivos y personalizados (Empático, Firme o Serio).

🔹 **Gestión de Cartera:** Puedo registrar, actualizar y eliminar clientes de tu base de datos.

🔹 **Consultas Rápidas:** Pídeme ver la lista de deudores cuando lo necesites.

**¿En qué puedo ayudarte hoy?**""",
        "bot"
    )
    
if __name__ == "__main__":
    ft.app(target=main)