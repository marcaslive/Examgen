# designer/views.py

import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST, require_GET
from django.contrib.auth.decorators import login_required

# Import the Manager from your service
from designer.services.question_generator import ExamGenerationManager
from designer.models import Document


def home(request):
    return render(request, "designer/home.html")


@login_required
def generate_exam_view(request):
    """Renders the generation HTML page."""
    docs = Document.objects.all().order_by("-id")
    return render(request, "designer/generate_exam.html", {"documents": docs})


@login_required
@require_POST
def generate_start_api(request):
    """Initializes generation, checks quotas, and builds the batch plan."""
    try:
        payload = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"success": False, "error": "Invalid JSON"}, status=400)

    status_code, response_data = ExamGenerationManager.start_generation(request, payload)
    return JsonResponse(response_data, status=status_code)


@login_required
@require_POST
def generate_batch_api(request):
    """Processes one chunk of questions (loop controlled by frontend)."""
    status_code, response_data = ExamGenerationManager.process_next_batch(request)
    return JsonResponse(response_data, status=status_code)


@login_required
@require_GET
def generate_quota_api(request):
    """Returns remaining RPM / RPD limits."""
    return JsonResponse({"success": True, **ExamGenerationManager.get_quota_status()})


# ---------------------------------------------------------
# IMPORTANT: PASTE YOUR OTHER VIEWS BELOW THIS LINE!
# (If you have admin_login_view, exam_list_view, etc., 
# they MUST be in this file so urls.py can find them!)
# ---------------------------------------------------------