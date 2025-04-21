import json
from django.views import View
from TestApp.path import logger
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from rest_framework.permissions import IsAuthenticated


@method_decorator(csrf_exempt, name='dispatch')  # Disable CSRF for this view
class PayloadView(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request, *args, **kwargs):
        try:
            logger.info("INFO: Successful POST request initiated")
            test_data = request.POST.get('test')  # Safely extract the 'test' key

            # Respond with success
            logger.info("INFO: Successful POST request initiated")
            return JsonResponse({
                "status": "success",
                "Test_Payload": test_data,
            }, status=200)
        except json.JSONDecodeError:
            logger.error(e)
            return JsonResponse({"status": "error", "message": "Invalid JSON"}, status=400)
        except KeyError as e:
            return JsonResponse({"status": "error", "message": f"Missing key: {str(e)}"}, status=400)
        except Exception as e:
            return JsonResponse({"status": "error", "message": f"Unexpected error: {str(e)}"}, status=500)

    def get(self, request, *args, **kwargs):
        # Optional: Handle GET requests if needed
        return JsonResponse({"status": "error", "message": "Only POST requests are allowed"}, status=405)
