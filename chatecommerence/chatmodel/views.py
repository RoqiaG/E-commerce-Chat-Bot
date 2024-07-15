from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
from chatmodel.utils.utils_model import handle_contextless_queries
from .utils.utils import load_model_from_h5, chat
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()


with open('chatbot/intents.json') as file:
    intents = json.load(file)
words = np.load('chatbot/words.pkl', allow_pickle=True)
classes = np.load('chatbot/classes.pkl', allow_pickle=True)
model_file_path = 'chatbot/chatbot_model.h5'

# def predict_view_chatbot(request):
#     if request.method == 'POST':
#         user_input = request.POST.get('userInput', '')
#         try:
#             model = load_model_from_h5(model_file_path)
#         except FileNotFoundError:
#             return render(request, 'error.html', {'message': 'Model file not found.'})
#         except Exception as e:
#             return render(request, 'error.html', {'message': f'Error loading model: {str(e)}'})

#         try:
#             response = chat(user_input, model, words, classes, intents)
#         except Exception as e:
#             return render(request, 'error.html', {'message': f'Error during prediction: {str(e)}'})

#         return render(request, 'index.html', {'user_input': user_input, 'prediction': response})

#     return render(request, 'index.html')

@method_decorator(csrf_exempt, name='dispatch')
class PredictViewChatbot(View):
    def post(self, request, *args, **kwargs):
        try:
            data = json.loads(request.body.decode('utf-8'))
            user_input = data.get('userInput', '')
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        
        try:
            model = load_model_from_h5(model_file_path)
        except FileNotFoundError:
            return JsonResponse({'error': 'Model file not found.'}, status=500)
        except Exception as e:
            return JsonResponse({'error': f'Error loading model: {str(e)}'}, status=500)

        try:
           # Call handle_contextless_queries to get the chatbot response
           response = handle_contextless_queries(user_input)
        except Exception as e:
            return JsonResponse({'error': f'Error during prediction: {str(e)}'}, status=500)

        return JsonResponse({'user_input': user_input, 'prediction': response})

    def get(self, request, *args, **kwargs):
        return JsonResponse({'message': 'This endpoint only accepts POST requests.'}, status=405)
