from django.urls import path
from . import views

from TestApp.viewset.payloadtest_viewset import PayloadView
from TestApp.viewset.classification_viewset import ClassificationView
from TestApp.viewset.list_models_viewset import ListModelsView
from TestApp.viewset.multi_classification_viewset import MultiClassificationViewset


urlpatterns = [
    path('test_payload/', PayloadView.as_view(), name='test_payload'),
    path('classify/', ClassificationView.as_view(), name='classification'),
    path('list_models/', ListModelsView.as_view(), name='list_models'),
    path('multi_classification/', MultiClassificationViewset.as_view(), name='multi_classification'),

]  
