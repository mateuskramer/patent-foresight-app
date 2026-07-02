# processador.py
import os
import json
import psycopg2
from psycopg2 import extras
from dotenv import load_dotenv
from google import genai
from google.genai import types
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import threading

load_dotenv()

# CONFIGURAÇÕES
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'database': os.environ.get('DB_NAME', 'patents'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASS', ''),
    'port': int(os.environ.get('DB_PORT', 5432)),
}

# Removido sslmode=require para compatibilidade com localhost/local
if os.environ.get('DB_SSLMODE'):
    DB_CONFIG['sslmode'] = os.environ.get('DB_SSLMODE')

# Dicionário inteligente de categorias
SMART_DICTIONARY = {
    "Vehicle Systems": ["vehicle", "car", "automotive", "truck", "bus", "lorry", "terrestrial vehicle", "chassis", "bodywork", "frame"],
    "Propulsion & Engine": ["engine", "motor", "combustion", "cylinder", "piston", "fuel system", "exhaust", "transmission", "gearbox", "clutch", "drivetrain"],
    "Electric & Hybrid": ["electric motor", "battery", "ev", "hybrid", "rechargeable", "charging", "inverter", "stator", "rotor", "bms", "lithium"],
    "Driving Assistance & Automation": ["steering", "braking", "abs", "adas", "autonomous", "self-driving", "driverless", "cruise control", "lane", "parking"],
    "Sensors & Navigation": ["radar", "lidar", "camera", "ultrasonic", "gps", "gnss", "navigation", "mapping", "slam", "odometry"],
    "Safety & Interior": ["airbag", "seatbelt", "safety", "collision", "dashboard", "hvac", "lighting", "headlamp", "infotainment"],
    "Tires & Suspension": ["tire", "tyre", "wheel", "suspension", "shock absorber", "axle", "rim"],
    "Food Raw Materials": ["food", "beverage", "ingredient", "additive", "protein", "carbohydrate", "lipid", "vitamin", "extract", "essence"],
    "Food Processing": ["processing", "milling", "grinding", "blending", "mixing", "heating", "cooling", "cooking", "frying", "baking", "extrusion"],
    "Preservation & Bio": ["preservation", "shelf-life", "pasteurization", "fermentation", "enzyme", "microbiological", "antimicrobial", "sterilization"],
    "Packaging": ["packaging", "container", "bottle", "can", "wrap", "film", "sealing", "vacuum", "labeling", "package"],
    "Nutritional & Bioactive": ["nutritional", "probiotic", "supplement", "functional food", "fortified", "antioxidant", "flavonoid"],
    "Dairy & Meat Tech": ["milk", "dairy", "cheese", "meat", "poultry", "fish", "plant-based meat", "alternative protein"],
    "Quality & Safety": ["haccp", "quality control", "pathogen", "contaminant", "toxicity", "ph level", "moisture"],
    "Fibers & Yarns": ["fiber", "fibre", "yarn", "thread", "filament", "synthetic", "natural fiber", "cotton", "wool", "silk", "polyester", "nylon", "acrylic"],
    "Fabric Construction": ["fabric", "textile", "weaving", "knitting", "non-woven", "woven", "braided", "mesh", "cloth", "garment"],
    "Chemical Treatment": ["dyeing", "dye", "pigment", "printing", "finishing", "coating", "bleaching", "scouring", "impregnation"],
    "Advanced Materials": ["polymer", "composite", "carbon fiber", "nanofiber", "resin", "plastic", "elastomer", "aramid", "kevlar"],
    "Smart & Technical Textiles": ["smart textile", "e-textile", "conductive", "waterproof", "breathable", "flame retardant", "protective clothing"],
    "Apparel & Fashion": ["clothing", "apparel", "wearable", "footwear", "sewing", "stitching", "pattern", "tailoring"],
    "Automation & Control": ["automation", "control system", "plc", "scada", "hmi", "controller", "actuator", "valve", "pneumatic", "hydraulic"],
    "Robotics": ["robot", "robotic", "arm", "manipulator", "end-effector", "cobot", "agv", "drone"],
    "IoT & Digital": ["iot", "internet of things", "sensor", "data", "cloud", "digital twin", "wireless", "rfid", "monitoring"],
    "Manufacturing Processes": ["manufacturing", "production", "assembly", "machining", "tooling", "molding", "casting", "forging", "stamping"],
    "Additive & 3D": ["3d printing", "additive manufacturing", "prototyping", "layering", "sintering"],
    "Maintenance & Quality": ["maintenance", "predictive", "inspection", "vision system", "testing", "calibration"]
}

CATEGORIES_TEXT = "\n".join([
    f"- {category}\n  Keywords: {', '.join(keywords)}"
    for category, keywords in SMART_DICTIONARY.items()
])

PROMPT_TEMPLATE = """You are a patent classification expert.
Analyze the patent text and select ONLY the categories strongly supported by the patent content.
Avoid broad or weak matches — only include a category if the patent clearly and directly relates to it.

AVAILABLE CATEGORIES AND KEYWORDS:
{categories}

PATENT TEXT:
{text}

RULES:
- Return ONLY exact category names from the list above.
- A category must be strongly supported by the patent text to be included.
- Do NOT include categories with only weak or tangential connections.
- Return ONLY a JSON object — no markdown, no explanation, no preamble.
- If nothing matches, return {{"topics": []}}.

FORMAT:
{{"topics": [{{"topic": "Category Name"}}, {{"topic": "Another Category"}}]}}"""

# ─────────────────────────────────────────────────────────────
# CONTROLE DE EXECUÇÃO (Dash-safe)
# ─────────────────────────────────────────────────────────────
PROCESSING_STATE = {
    "running": False,
    "stop_requested": False,
    "logs": [],
    "thread": None
}

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=15),
    retry=retry_if_exception_type(Exception)
)
def classify_with_retry(client, text):
    prompt = PROMPT_TEMPLATE.format(categories=CATEGORIES_TEXT, text=text)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            top_p=0.1,
            max_output_tokens=2048,
        )
    )
    return response.text

def _run_processor_in_thread():
    """Função interna que roda em background para não travar o Dash."""
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=extras.DictCursor)

        cur.execute("""
            SELECT p.id, p.title, p.abstract 
            FROM patents p
            LEFT JOIN patent_terms pt ON p.id = pt.patent_id
            WHERE pt.patent_id IS NULL
            ORDER BY p.id ASC
            LIMIT 50
        """)
        patentes = cur.fetchall()
        conn.close()

        if not patentes:
            PROCESSING_STATE["logs"].append("📭 Sem patentes novas para analisar.")
            PROCESSING_STATE["running"] = False
            return

        PROCESSING_STATE["logs"].append(f"🚀 Iniciando processamento de {len(patentes)} patentes...")

        for idx, pat in enumerate(patentes):
            if PROCESSING_STATE["stop_requested"]:
                PROCESSING_STATE["logs"].append("🛑 Interrupção solicitada pelo usuário.")
                break

            PROCESSING_STATE["logs"].append(f"🧠 [{idx+1}/{len(patentes)}] Analisando ID: {pat['id']}")
            text = f"{pat['title']}. {pat['abstract']}"

            try:
                raw = classify_with_retry(client, text)
                clean = raw.replace('```json', '').replace('```', '').strip()
                data = json.loads(clean)
                topics = data.get("topics", [])

                conn = psycopg2.connect(**DB_CONFIG)
                cur = conn.cursor()

                for item in topics:
                    term = item['topic'].strip().lower()
                    if not term: continue
                    
                    cur.execute("""
                        INSERT INTO term_dictionary (term, class, status)
                        VALUES (%s, 'technology', 'approved')
                        ON CONFLICT (term) DO NOTHING RETURNING id
                    """, (term,))
                    res_id = cur.fetchone()
                    term_id = res_id[0] if res_id else None

                    if not term_id:
                        cur.execute("SELECT id FROM term_dictionary WHERE term = %s", (term,))
                        row = cur.fetchone()
                        term_id = row[0] if row else None

                    if term_id:
                        cur.execute(
                            "INSERT INTO patent_terms (patent_id, term_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                            (pat['id'], term_id)
                        )

                conn.commit()
                conn.close()
                PROCESSING_STATE["logs"].append(f"✅ ID {pat['id']} processado ({len(topics)} termos).")
                time.sleep(15)

            except Exception as e:
                if conn: conn.rollback()
                PROCESSING_STATE["logs"].append(f"⚠️ Erro no ID {pat['id']}: {str(e)[:50]}...")
                time.sleep(5)

        PROCESSING_STATE["logs"].append("🏁 Fim do processamento.")
        PROCESSING_STATE["running"] = False
        PROCESSING_STATE["stop_requested"] = False

    except Exception as e:
        PROCESSING_STATE["logs"].append(f"❌ Erro crítico: {str(e)}")
        PROCESSING_STATE["running"] = False
        PROCESSING_STATE["stop_requested"] = False

def start_processing():
    """Inicia o processamento em thread separada."""
    if PROCESSING_STATE["running"]:
        return "⚠️ Processamento já em andamento."
    PROCESSING_STATE["running"] = True
    PROCESSING_STATE["stop_requested"] = False
    PROCESSING_STATE["logs"] = []
    PROCESSING_STATE["thread"] = threading.Thread(target=_run_processor_in_thread, daemon=True)
    PROCESSING_STATE["thread"].start()
    return "🚀 Processamento iniciado em segundo plano."

def stop_processing():
    """Solicita parada suave."""
    if PROCESSING_STATE["running"]:
        PROCESSING_STATE["stop_requested"] = True
        return "🛑 Solicitação de parada enviada."
    return "⏸️ Nenhum processamento ativo."

def get_logs():
    """Retorna logs acumulados."""
    return list(PROCESSING_STATE["logs"])

def is_running():
    return PROCESSING_STATE["running"]