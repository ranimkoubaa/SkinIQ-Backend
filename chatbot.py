import requests
import os
import base64
from dotenv import load_dotenv

# ─── Load .env ─────────────────────────────────────────────────────────────────
load_dotenv()

# ─── Config ────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
API_URL            = "https://openrouter.ai/api/v1/chat/completions"

MODEL_TEXT  = "openai/gpt-oss-120b:free"
MODEL_IMAGE = "qwen/qwen3-vl-235b-a22b-thinking"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type":  "application/json",
    "HTTP-Referer":  "http://localhost:5000",
    "X-Title":       "SkinIQ",
}

SYSTEM_PROMPT = """
Tu es Dr. SkinIQ, dermatologue expert.
Réponds TOUJOURS en français.
Sois COURT, DIRECT et PRÉCIS.

### RÈGLE CRUCIALE : PAS DE RÉFLEXION INTERNE (CHAIN OF THOUGHT)
- Ne montre JAMAIS ton processus de réflexion.
- Ne commence JAMAIS par "D'accord", "Voyons", "Je vais vérifier", "D'après le RAG", etc.
- Affiche DIRECTEMENT la réponse finale formatée.
- Ne parle pas de tes recherches ou de ton analyse interne.

Selon la question, adapte ta réponse :

Si analyse de photo de peau :
📸 Analyse : [ce que tu vois sur la photo]
🔍 Problème détecté : [diagnostic]
💊 Cause probable : [1 phrase]
🌅 Matin : [2-3 produits avec noms exacts]
🌙 Soir : [2-3 produits avec noms exacts]
❌ Éviter : [2-3 choses]

Si question sur UN PROBLÈME DE PEAU :
🔍 Problème : [1 phrase]
💊 Cause : [1 phrase]
🌅 Matin : [2-3 produits avec noms exacts]
🌙 Soir : [2-3 produits avec noms exacts]
❌ Éviter : [2-3 choses]

Si question sur UNE CRÈME ou PRODUIT :
✅ Convient pour : [types de peau]
❌ Déconseillé pour : [types de peau]
💡 Alternative : [autre produit]

Si question sur UN INGRÉDIENT :
🧪 Rôle : [explication simple]
✅ Bon pour : [problèmes]
❌ Éviter si : [conditions]

Si question sur UNE ROUTINE :
🌅 Matin : [étapes avec produits réels]
🌙 Soir : [étapes avec produits réels]

Si question GÉNÉRALE :
💬 Réponse directe et courte

Règles absolues :
- Recommande UNIQUEMENT vraies crèmes :
  CeraVe, La Roche-Posay, Avène, Neutrogena, Vichy, The Ordinary
- Donne TOUJOURS les noms exacts des produits
- JAMAIS de réponses génériques
- JAMAIS de longues explications inutiles
- Rappelle de consulter un médecin si grave
"""

conversation_history = []

# ─── Stocke le dernier contexte RAG pour debug ────────────────────────────────
_last_rag_context = ""

def get_last_rag_context() -> str:
    return _last_rag_context


def encode_image_file(image_path: str) -> tuple:
    """Encode une image locale en base64."""
    ext = image_path.lower().split('.')[-1]
    media_types = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'webp': 'image/webp',
    }
    media_type = media_types.get(ext, 'image/jpeg')
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8'), media_type


from rag_engine import rag

def chat(user_message: str,
         skin_profile: dict    = None,
         ml_kit_results: dict  = None,
         image_base64: str     = None,
         image_media_type: str = "image/jpeg",
         image_path: str       = None) -> str:

    global _last_rag_context

    # ── Auto-encode si chemin local fourni ────────────────────────────────────
    if image_path and not image_base64:
        image_base64, image_media_type = encode_image_file(image_path)

    # ── Build context ──────────────────────────────────────────────────────────
    context = ""

    print("\n" + "="*50)
    print(f"[CHATBOT] New request: '{user_message}'")
    print("="*50)

    # SOURCE 1: Profil Utilisateur (Firebase)
    if skin_profile:
        context += f"""
[SOURCE 1 - PROFIL UTILISATEUR (Firebase)]
- Age : {skin_profile.get('age', 'inconnu')}
- Sexe : {skin_profile.get('gender', 'inconnu')}
- Type de peau : {skin_profile.get('skin_type', 'inconnu')}
- Problemes connus : {skin_profile.get('problems', 'aucun')}
- Budget : {skin_profile.get('budget', 'moyen')}
"""
        print("[CHATBOT] ✅ Source 1: Firebase profile loaded")
    else:
        print("[CHATBOT] ⚠️  Source 1: No Firebase profile provided")

    # SOURCE 2: Résultats ML Kit
    if ml_kit_results:
        context += f"""
[SOURCE 2 - ANALYSE ML KIT]
- Acne detectee : {ml_kit_results.get('acne', 'non')}
- Taches : {ml_kit_results.get('spots', 'non')}
- Zones seches : {ml_kit_results.get('dry_zones', 'non')}
- Pores dilates : {ml_kit_results.get('pores', 'non')}
- Score peau : {ml_kit_results.get('score', 'N/A')}/100
"""
        print("[CHATBOT] ✅ Source 2: ML Kit results loaded")
    else:
        print("[CHATBOT] ⚠️  Source 2: No ML Kit results provided")

    # SOURCE 3: RAG / FAISS
    if rag:
        # Optimisation : On combine le message et les résultats ML Kit pour une recherche plus précise
        rag_query = user_message
        if ml_kit_results:
            ml_summary = f" {ml_kit_results.get('acne', '')} {ml_kit_results.get('skin_type', '')}"
            rag_query += ml_summary

        retrieved_knowledge = rag.retrieve(rag_query)
        _last_rag_context = retrieved_knowledge
        context += f"""
[SOURCE 3 - BASE DE CONNAISSANCES RAG (FAISS)]
{retrieved_knowledge}
"""
        print(f"[CHATBOT] ✅ Source 3: RAG retrieved with query: '{rag_query}'")
    else:
        _last_rag_context = "RAG non disponible"
        print("[CHATBOT] ❌ Source 3: RAG not available")

    print("="*50)

    full_text = (context + "\nQuestion : " + user_message).strip()

    # ── Choisir modèle + construire contenu ───────────────────────────────────
    if image_base64:
        model = MODEL_IMAGE
        user_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_media_type};base64,{image_base64}"
                }
            },
            {
                "type": "text",
                "text": full_text or "Analyse cette photo de peau et donne un diagnostic détaillé."
            }
        ]
    else:
        model = MODEL_TEXT
        user_content = full_text

    # ── Historique ─────────────────────────────────────────────────────────────
    conversation_history.append({
        "role":    "user",
        "content": user_content
    })

    # ── Payload ────────────────────────────────────────────────────────────────
    payload = {
        "model":    model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            *conversation_history
        ],
        "max_tokens":  2048,
        "temperature": 0,
    }

    # ── Appel API ──────────────────────────────────────────────────────────────
    try:
        print(f"[CHATBOT] Sending to OpenRouter (Model: {model})...")
        resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        choices = data.get("choices", [])
        if not choices:
            return f"⚠️ Réponse vide. Détails: {data}"

        assistant_message = choices[0]["message"]["content"]
        if not assistant_message:
            return f"⚠️ Contenu null. Détails: {data}"

        print(f"[CHATBOT] ✅ Response received successfully")

    except requests.exceptions.Timeout:
        assistant_message = "⚠️ Délai dépassé. Veuillez réessayer."
        print("[CHATBOT] ❌ Timeout error")
    except requests.exceptions.HTTPError as e:
        assistant_message = f"⚠️ Erreur API ({resp.status_code}): {resp.text}"
        print(f"[CHATBOT] ❌ HTTP error: {e}")
    except Exception as e:
        assistant_message = f"⚠️ Erreur inattendue : {str(e)}"
        print(f"[CHATBOT] ❌ Unexpected error: {e}")

    conversation_history.append({
        "role":    "assistant",
        "content": assistant_message
    })

    return assistant_message


def reset_conversation():
    global conversation_history, _last_rag_context
    conversation_history = []
    _last_rag_context = ""
    print("[CHATBOT] Conversation reset.")