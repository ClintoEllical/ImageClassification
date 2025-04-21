import os
import json
from django.views import View
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework.permissions import IsAuthenticated
from TestApp.services.classification_service import classify_images
from TestApp.path import MODELS_DIR,DATASET_DIR




from TestApp.path import logger

@method_decorator(csrf_exempt, name='dispatch')
class ClassificationView(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request, *args, **kwargs):
        try:
            logger.info("INFO : Classification service started")
            # Get form data from POST request
            model_name      = request.POST.get('model_name')
            dataset_folder  = request.POST.get('dataset')

            # Validate form data
            if not model_name or not dataset_folder:
                return JsonResponse({
                    "status": "error",
                    "message": "Both 'model_name' and 'dataset_path' are required."
                }, status=400)

            # Call classification service
            results_path = classify_images(model_name,dataset_folder)

            logger.info("INFO : Classification service ended")
            # Return success response
            return JsonResponse({
                "status": "success",
                "message": "Classification complete.",
                "results_path": results_path
            }, status=200)
        
        except Exception as e:
            return JsonResponse({"status": "error", "message": f"Unexpected error: {str(e)}"})
