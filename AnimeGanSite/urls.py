"""AnimeGanSite URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from login import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('index/', views.index),
    path('login/', views.login),
    path('enroll/', views.enroll),
    path('home/', views.home),
    path('home/exit/', views.user_exit),
    path('home/list/', views.img_list),
    path('upload/', views.upload),
    path('home/del/', views.img_del),
    path('home/down/', views.img_down),
    path('home/view/', views.img_view),
    path('manage/login/', views.manage_login),
    path('user/list/', views.user_list),
    path('user/del/', views.user_del),
]
