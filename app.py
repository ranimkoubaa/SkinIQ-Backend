from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import chat, reset_conversation, get_last_rag_context
from rag_engine import rag

app = Flask(__name__)
CORS(app)


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        data             = request.json
        user_message     = data.get('message', '')
        skin_profile     = data.get('skin_profile', None)
        ml_kit_results   = data.get('ml_kit_results', None)
        image_base64     = data.get('image_base64', None)
        image_media_type = data.get('image_media_type', 'image/jpeg')

        if not user_message and not image_base64:
            return jsonify({'error': 'Message ou image_base64 requis'}), 400

        response = chat(
            user_message,
            skin_profile,
            ml_kit_results,
            image_base64,
            image_media_type,
        )

        # ── RAG Debug Info ─────────────────────────────────────────────────────
        rag_context = get_last_rag_context()

        return jsonify({
            'status':   'success',
            'response': response,
            # RAG debug info
            'rag_debug': {
                'rag_used': True,
                'retrieval_sources': {
                    'firebase_profile': skin_profile is not None,
                    'mlkit_results':    ml_kit_results is not None,
                    'vector_db':        rag is not None,
                },
                'retrieved_documents': rag_context,
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset_endpoint():
    reset_conversation()
    return jsonify({'status': 'conversation reinitialisee'})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'SkinIQ backend running ⚡',
        'rag_ready': rag is not None,
        'rag_documents': len(rag.documents) if rag else 0,
        'model': 'openai/gpt-oss-120b:free'
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)