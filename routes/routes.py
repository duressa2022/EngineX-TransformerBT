from models.model import TranslationRequest,TranslationResponse
from service.service import translate_fn
from flask import request,Blueprint,jsonify

translation_bp=Blueprint('translation_bp',__name__)

@translation_bp.route('/translate',methods=['POST'])
def translate():
    print(request.json())
    try:
        data=request.json()
        print("data: ",data)
        if not data:
            return jsonify({
                "message":"error from client"
            }),400
        
        request=TranslationRequest.from_dict(data=data)
        translation=translate_fn(request.sentence)
        response=TranslationResponse(translation=translation)

        return jsonify({
            response.to_dict()
        }),200
    except ValueError as ve:
        return jsonify({
            "message":str(ve),
            "error":"value error"
        }),400
    except Exception as e:
        return jsonify({
            "message":str(e),
            "error":"server error"
        }),500
    
        