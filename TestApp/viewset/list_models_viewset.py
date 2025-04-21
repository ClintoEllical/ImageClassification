import json
import os
from django.views import View
from TestApp.path import logger
from importlib import import_module
from TestApp.path import MODELS_DIR
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework.permissions import IsAuthenticated


@method_decorator(csrf_exempt, name='dispatch')  # Disable CSRF for this view
class ListModelsView(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request, *args, **kwargs):
        try:
            logger.info("INFO : List of models service started")
            path= MODELS_DIR
            models_path = MODELS_DIR
            model_names = []
            for root, _, files in os.walk(path):
                for file in files:
                    if file != "__init__.py":
                        model_names.append(file)     
            logger.info("INFO : List of models service completed")  
            return JsonResponse({
                    "status": "success",
                    "Test_Payload": model_names,
                }, status=200)              
        except json.JSONDecodeError:
            return JsonResponse({"status": "error", "message": "Invalid JSON"}, status=400)
        except KeyError as e:
            return JsonResponse({"status": "error", "message": f"Missing key: {str(e)}"}, status=400)
        except Exception as e:
            return JsonResponse({"status": "error", "message": f"Unexpected error: {str(e)}"}, status=500)

    def get(self, request, *args, **kwargs):
        # Optional: Handle GET requests if needed
        return JsonResponse({"status": "error", "message": "Only POST requests are allowed"}, status=405)

