import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from langchain_core.messages import SystemMessage, HumanMessage

# Importation depuis nos propres modules
from config import UPLOAD_FOLDER
from database import vector_store
from document_processor import process_and_add_file
from rag_agent import rag_agent

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/add_file', methods=['POST'])
def add_file():
    """API pour ajouter un nouveau fichier PDF pour un patient."""
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier fourni"}), 400
        
    file = request.files['file']
    patient_id = request.form.get('patient_id')
    file_id = request.form.get('file_id')
    
    if not patient_id or not file_id or file.filename == '':
        return jsonify({"error": "Paramètres manquants (file, patient_id, file_id)"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        num_chunks = process_and_add_file(file_path, patient_id, file_id)
        return jsonify({
            "status": "success",
            "message": f"{num_chunks} segments ajoutés avec succès pour le fichier '{file_id}'."
        }), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/update_file', methods=['POST'])
def update_file():
    """API pour mettre à jour un fichier existant."""
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier fourni"}), 400
        
    file = request.files['file']
    patient_id = request.form.get('patient_id')
    file_id = request.form.get('file_id')
    
    if not patient_id or not file_id or file.filename == '':
        return jsonify({"error": "Paramètres manquants"}), 400

    # Supprimer les anciens segments liés à ce file_id
    existing_docs = vector_store.get(where={"file_id": file_id})
    deleted_count = 0
    if existing_docs and existing_docs['ids']:
        deleted_count = len(existing_docs['ids'])
        vector_store.delete(ids=existing_docs['ids'])

    # Sauvegarder et traiter le nouveau fichier
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    try:
        num_chunks = process_and_add_file(file_path, patient_id, file_id)
        return jsonify({
            "status": "success",
            "message": f"Mise à jour réussie. {deleted_count} anciens segments supprimés. {num_chunks} nouveaux segments ajoutés pour '{file_id}'."
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/query', methods=['POST'])
def query_patient():
    """API pour poser une question sur les données d'un patient."""
    data = request.json
    if not data or 'patient_id' not in data or 'question' not in data:
        return jsonify({"error": "Le corps de la requête doit contenir 'patient_id' et 'question'"}), 400
        
    patient_id = data['patient_id']
    question = data['question']
    
    sys_prompt = SystemMessage(
        content=f"You are a medical AI assistant. You are currently assisting with patient_id: '{patient_id}'. "
                "Always use the retrieve_patient_records tool to look up information before answering. "
                "Be concise and base your answers strictly on the context provided."
    )
    user_msg = HumanMessage(content=question)
    
    try:
        inputs = {"messages": [sys_prompt, user_msg]}
        final_state = rag_agent.invoke(inputs)
        final_message = final_state["messages"][-1]
        
        return jsonify({
            "patient_id": patient_id,
            "question": question,
            "answer": final_message.content
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)