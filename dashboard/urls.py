from django.urls import path
from .views import index, train_stream

urlpatterns = [
    path('', index),
    path('train/', train_stream),
]
