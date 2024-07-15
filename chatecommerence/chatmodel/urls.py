from django.urls import path
from .views import PredictViewChatbot

urlpatterns = [
    path('predict/', PredictViewChatbot.as_view(), name='predict'),
]
