from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np
import os
import sys

class RAGEngine:
    def __init__(self, knowledge_path='skincare_knowledge.json'):
        self.knowledge_path = knowledge_path
        self.model = None
        self.index = None
        self.documents = []
        
        print("\n" + "="*50)
        print("[RAG] Initializing RAG Engine...")
        print("="*50)

        # 1. Load Model
        try:
            print("[RAG] Step 1: Loading SentenceTransformer model...")
            # paraphrase-multilingual-MiniLM-L12-v2 is good for French
            self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("[RAG] Model loaded successfully.")
        except Exception as e:
            print(f"[RAG] ERROR: Could not load model. {str(e)}")
            # We don't exit here to allow the app to potentially run without RAG if needed, 
            # but in this context, it's better to know why it failed.
            return

        # 2. Load Knowledge Base
        try:
            print(f"[RAG] Step 2: Loading knowledge base from {knowledge_path}...")
            if not os.path.exists(knowledge_path):
                raise FileNotFoundError(f"File not found: {knowledge_path}")
                
            with open(knowledge_path, 'r', encoding='utf-8') as f:
                self.knowledge = json.load(f)
            print(f"[RAG] Knowledge base loaded ({len(self.knowledge.get('ingredients', []))} ingredients, {len(self.knowledge.get('conditions', []))} conditions).")
        except FileNotFoundError as e:
            print(f"[RAG] ERROR: {str(e)}")
            return
        except json.JSONDecodeError as e:
            print(f"[RAG] ERROR: Failed to parse JSON. {str(e)}")
            return
        except Exception as e:
            print(f"[RAG] ERROR: Unexpected error loading knowledge. {str(e)}")
            return

        # 3. Prepare Documents
        print("[RAG] Step 3: Preparing documents for indexing...")
        self._prepare_documents()
        print(f"[RAG] {len(self.documents)} documents prepared.")

        # 4. Build Index
        try:
            print("[RAG] Step 4: Building FAISS index...")
            self._build_index()
            print("[RAG] FAISS index built successfully.")
        except Exception as e:
            print(f"[RAG] ERROR: Failed to build index. {str(e)}")
            return

        print("="*50)
        print(f"[RAG] RAG Engine is READY with {len(self.documents)} documents.")
        print("="*50 + "\n")

    def _prepare_documents(self):
        # Ingredients
        for item in self.knowledge.get('ingredients', []):
            doc = (
                f"Ingrédient {item['name']} : {item['role']}. "
                f"Recommandé pour : {', '.join(item['good_for'])}. "
                f"À éviter si : {', '.join(item['bad_for'])}. "
                f"Exemples de produits : {', '.join(item.get('products', []))}"
            )
            self.documents.append(doc)
        
        # Conditions
        for item in self.knowledge.get('conditions', []):
            doc = (
                f"Problème/Condition {item['name']} : {item['cause']}. "
                f"Traitement recommandé : {item['treatment']}. "
                f"À éviter : {item['avoid']}. "
                f"Produits recommandés : {', '.join(item.get('recommended_products', []))}"
            )
            self.documents.append(doc)
        
        # Routines
        for item in self.knowledge.get('routines', []):
            doc = (
                f"Routine pour type de peau {item['skin_type']} : "
                f"Matin : {', '.join(item['morning'])}. "
                f"Soir : {', '.join(item['evening'])}"
            )
            self.documents.append(doc)

    def _build_index(self):
        if not self.documents:
            print("[RAG] WARNING: No documents to index.")
            return

        embeddings = self.model.encode(self.documents)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

    def retrieve(self, query: str, top_k: int = 3) -> str:
        if not self.index or not self.model:
            print("[RAG] ERROR: Retrieval failed because engine is not properly initialized.")
            return "Aucune connaissance supplémentaire disponible (moteur RAG non initialisé)."

        print(f"[RAG] Retrieving context for query: '{query}'")
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.documents):
                results.append(f"• {self.documents[idx]}")
        
        if not results:
            print("[RAG] No relevant documents found.")
            return "Aucune connaissance spécifique trouvée."
            
        print(f"[RAG] Found {len(results)} relevant documents.")
        return "\n".join(results)

# Create a singleton instance
try:
    rag = RAGEngine()
except Exception as e:
    print(f"[RAG] CRITICAL: Failed to create RAG instance: {str(e)}")
    rag = None

if __name__ == "__main__":
    # Test script
    if rag:
        test_query = "Quels sont les bienfaits de la niacinamide ?"
        print(f"\n[TEST] Query: {test_query}")
        print("[TEST] Result:")
        print(rag.retrieve(test_query))
        
        test_query_2 = "J'ai la peau grasse avec de l'acné"
        print(f"\n[TEST] Query: {test_query_2}")
        print("[TEST] Result:")
        print(rag.retrieve(test_query_2))
    else:
        print("[TEST] RAG instance not available.")
