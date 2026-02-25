from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.models import User
from .forms import DoctorRegistrationForm, UserRegistrationForm, UserProfileForm, DoctorProfileForm, AppointmentForm
from .models import DoctorProfile, UserProfile, Appointment
from prediction.models import Scan

@user_passes_test(lambda u: u.is_superuser)
def admin_dashboard_view(request):
    doctors = DoctorProfile.objects.all().select_related('user').order_by('-created_at')
    
    context = {
        'doctors': doctors,
        'total_users': UserProfile.objects.count(),
        'pending_doctors': doctors.filter(is_approved=False).count(),
        'approved_doctors': doctors.filter(is_approved=True).count()
    }
    return render(request, 'accounts/admin_dashboard.html', context)

@user_passes_test(lambda u: u.is_superuser)
def all_users_list_view(request):
    users = UserProfile.objects.all().select_related('user').order_by('-created_at')
    return render(request, 'accounts/admin_users.html', {'users': users})

@user_passes_test(lambda u: u.is_superuser)
def approve_doctor_view(request, pk):
    doctor = get_object_or_404(DoctorProfile, pk=pk)
    doctor.is_approved = True
    doctor.save()
    messages.success(request, f"Dr. {doctor.user.last_name} has been approved.")
    return redirect('accounts:admin_dashboard')

def register_view(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST, request.FILES)
        if form.is_valid():
            user = form.save()
            messages.success(request, f"Welcome to DermoAI, {user.username}! Please login to continue.")
            return redirect('accounts:login')
    else:
        form = UserRegistrationForm()
    return render(request, 'accounts/register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.info(request, f"You are now logged in as {username}.")
                if hasattr(user, 'doctor_profile'):
                    return redirect('accounts:dr_dashboard')
                return redirect('home')
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()
    return render(request, 'accounts/login.html', {'form': form})

def logout_view(request):
    logout(request)
    messages.info(request, "You have successfully logged out.")
    return redirect('home')

def dr_register_view(request):
    if request.method == 'POST':
        form = DoctorRegistrationForm(request.POST, request.FILES)
        if form.is_valid():
            user = form.save()
            messages.success(request, f"Registration successful! Welcome, Dr. {user.last_name}. Your account is pending administrative approval. Please wait for verification before logging in.")
            return redirect('accounts:dr_login')
    else:
        form = DoctorRegistrationForm()
    return render(request, 'accounts/dr_register.html', {'form': form})

def dr_login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                if hasattr(user, 'doctor_profile'):
                    if user.doctor_profile.is_approved:
                        login(request, user)
                        messages.info(request, f"Welcome back, Dr. {user.last_name}.")
                        return redirect('accounts:dr_dashboard')
                    else:
                        messages.warning(request, "Your account is still pending administrative approval. Please try again later.")
                        return redirect('accounts:dr_login')
                else:
                    messages.error(request, "This account is not registered as a doctor.")
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    else:
        form = AuthenticationForm()
    return render(request, 'accounts/dr_login.html', {'form': form})

def dr_dashboard_view(request):
    if not hasattr(request.user, 'doctor_profile'):
        messages.error(request, "Access denied. Doctor profile required.")
        return redirect('home')
    
    scans = Scan.objects.all().order_by('-created_at')
    appointments = request.user.doctor_profile.appointments.all().order_by('-date', '-time')
    
    context = {
        'scans': scans,
        'doctor': request.user.doctor_profile,
        'appointments': appointments,
        'total_scans': scans.count(),
        'high_risk': scans.filter(risk_level='High').count(),
        'pending_appt': appointments.filter(status='Pending').count()
    }
    return render(request, 'accounts/dr_dashboard.html', context)

@login_required
def doctor_list_view(request):
    doctors = DoctorProfile.objects.filter(is_approved=True).select_related('user')
    return render(request, 'accounts/doctor_list.html', {'doctors': doctors})

@login_required
def book_appointment_view(request, dr_id):
    doctor = get_object_or_404(DoctorProfile, id=dr_id)
    if request.method == 'POST':
        form = AppointmentForm(request.POST)
        if form.is_valid():
            appointment = form.save(commit=False)
            appointment.patient = request.user
            appointment.doctor = doctor
            appointment.save()
            messages.success(request, f"Appointment request sent to Dr. {doctor.user.last_name}. Please wait for approval.")
            return redirect('home')
    else:
        form = AppointmentForm()
    
    return render(request, 'accounts/book_appointment.html', {
        'form': form,
        'doctor': doctor
    })

@login_required
def update_appointment_status(request, appt_id, status):
    if not hasattr(request.user, 'doctor_profile'):
        messages.error(request, "Access denied.")
        return redirect('home')
    
    appointment = get_object_or_404(Appointment, id=appt_id, doctor=request.user.doctor_profile)
    if status in ['Approved', 'Cancelled', 'Completed']:
        appointment.status = status
        appointment.save()
        messages.success(request, f"Appointment status updated to {status}.")
    
    return redirect('accounts:dr_dashboard')

@login_required
def profile_view(request):
    profile = request.user.profile
    if request.method == 'POST':
        form = UserProfileForm(request.POST, request.FILES, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, "Your profile has been updated successfully.")
            return redirect('accounts:profile')
    else:
        form = UserProfileForm(instance=profile)
    
    return render(request, 'accounts/profile.html', {
        'profile': profile,
        'form': form
    })

@login_required
def dr_profile_view(request):
    if not hasattr(request.user, 'doctor_profile'):
        messages.error(request, "Access denied. Doctor profile required.")
        return redirect('home')
    
    profile = request.user.doctor_profile
    if request.method == 'POST':
        form = DoctorProfileForm(request.POST, request.FILES, instance=profile)
        if form.is_valid():
            form.save()
            messages.success(request, "Your professional profile has been updated.")
            return redirect('accounts:dr_profile')
    else:
        form = DoctorProfileForm(instance=profile)
        
    return render(request, 'accounts/dr_profile.html', {
        'profile': profile,
        'form': form
    })
