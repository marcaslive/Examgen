from django.contrib import admin
from django.contrib.auth import views as auth_views
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.views.decorators.csrf import csrf_exempt
import os


# Check if we're in preview mode
IS_PREVIEW = os.environ.get("E2B_SANDBOX", "").lower() == "true"


# Custom admin login that redirects to admin index
def admin_login_view(request):
    """Custom admin login that redirects to django-admin after success."""
    from django.contrib.auth.views import LoginView
    from django.urls import reverse
    
    class AdminLoginView(LoginView):
        template_name = 'admin/login.html'
        
        def get_success_url(self):
            return reverse('admin:index')
    
    view = AdminLoginView.as_view()
    # Exempt CSRF in preview mode (iframe cookie blocking)
    if IS_PREVIEW:
        view = csrf_exempt(view)
    return view(request)


urlpatterns = [
    path('django-admin/login/', admin_login_view, name='admin_login_custom'),
    path('django-admin/', admin.site.urls),
    path('', include(('designer.urls', 'designer'), namespace='designer')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)