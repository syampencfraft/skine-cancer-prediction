from django.urls import path
from . import views

app_name = 'accounts'

urlpatterns = [
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('dr-register/', views.dr_register_view, name='dr_register'),
    path('dr-login/', views.dr_login_view, name='dr_login'),
    path('dr-dashboard/', views.dr_dashboard_view, name='dr_dashboard'),
    path('admin-dashboard/', views.admin_dashboard_view, name='admin_dashboard'),
    path('admin-users/', views.all_users_list_view, name='all_users_list'),
    path('approve-doctor/<int:pk>/', views.approve_doctor_view, name='approve_doctor'),
    path('profile/', views.profile_view, name='profile'),
    path('dr-profile/', views.dr_profile_view, name='dr_profile'),
    path('find-doctor/', views.doctor_list_view, name='doctor_list'),
    path('book-appointment/<int:dr_id>/', views.book_appointment_view, name='book_appointment'),
    path('update-appointment/<int:appt_id>/<str:status>/', views.update_appointment_status, name='update_appointment'),
]
