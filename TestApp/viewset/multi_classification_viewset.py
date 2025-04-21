import json
import requests
from django.views import View
from TestApp.path import logger,localhost
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework.permissions import IsAuthenticated


@method_decorator(csrf_exempt, name='dispatch')  # Disable CSRF for this view
class MultiClassificationViewset(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request, *args, **kwargs):
        try:
            logger.info("INFO: MultiClassificationViewset started")
            num_models      = int(request.POST.get('models_num', 0))  # Default to 0 if not provided
            models_payload  = request.POST.get('model_names')  # Example: "blur_classify.h5,auto_classify.h5"
            
            if not models_payload:
                return JsonResponse({
                    "status": "error",
                    "message": "No model names provided."
                }, status=400)

            # Parse the model names into a list
            model_names = [name.strip() for name in models_payload.split(",")]
            results = {}

            logger.info("INFO: MultiClassificationViewset hitting classification API")
            for model_name in model_names:
                data = {"model_name": model_name,     
                        "dataset": "blur_test"}
                try:
                    response = requests.post(localhost, data=data)  # Adjust URL accordingly
                    results[model_name] = response.json()  # Store response
                except requests.exceptions.RequestException as e:
                    results[model_name] = {"status": "error", "message": str(e)}

            logger.info("INFO: MultiClassificationViewset ended")
            # Respond with success
            return JsonResponse({
                "status": "success",
                "num_models": num_models,
                "model_names": model_names,
                "results": results,
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

