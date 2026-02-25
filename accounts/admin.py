from django.contrib import admin
from .models import UserProfile, DoctorProfile

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'age', 'gender', 'created_at')
    search_fields = ('user__username', 'user__email')

@admin.register(DoctorProfile)
class DoctorProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'specialization', 'is_approved', 'created_at')
    list_filter = ('is_approved', 'specialization')
    search_fields = ('user__username', 'user__email', 'hospital_name')
    actions = ['approve_doctors']

    def approve_doctors(self, request, queryset):
        queryset.update(is_approved=True)
    approve_doctors.short_description = "Mark selected doctors as approved"
