# designer/urls.py

from django.urls import path
from . import views
app_name = 'designer'
urlpatterns = [

    # ─── Home ───────────────────────────────────────
    path('', views.home, name='home'),

    # ─── Authentication ─────────────────────────────
    path('admin/login/', views.admin_login_view, name='admin_login'),
    path('login/', views.user_login_view, name='user_login'),
    path('signup/', views.user_signup_view, name='signup'),
    path('logout/', views.logout_view, name='logout'),

    # ─── Dashboards ─────────────────────────────────
    path('admin-dashboard/', views.admin_dashboard_view, name='admin_dashboard'),
    path('user-dashboard/', views.user_dashboard_view, name='user_dashboard'),

    # ─── Documents ──────────────────────────────────
    path('documents/', views.documents_view, name='documents'),
    path('documents/upload/', views.document_upload_view, name='document_upload'),
    path('documents/<uuid:doc_id>/delete/', views.document_delete_view, name='document_delete'),
    path('documents/reset/', views.documents_reset_view, name='documents_reset'),

    # ─── Exam Generation ────────────────────────────
    path('generate/', views.generate_exam_view, name='generate_exam'),
    path('generate/questions/', views.generate_questions_api, name='generate_questions'),
    path('generate/review/', views.review_questions_view, name='review_questions'),
    path('generate/save/', views.save_exam_view, name='save_exam'),

    # ─── Exams ──────────────────────────────────────
    path('exams/', views.exam_list_view, name='exam_list'),
    path('exams/<uuid:exam_id>/', views.exam_detail_view, name='exam_detail'),
    path('exams/<uuid:exam_id>/edit/', views.exam_edit_view, name='exam_edit'),
    path('exams/<uuid:exam_id>/delete/', views.exam_delete_view, name='exam_delete'),
    path('exams/<uuid:exam_id>/publish/', views.exam_publish_view, name='exam_publish'),
    path('exams/<uuid:exam_id>/duplicate/', views.exam_duplicate_view, name='exam_duplicate'),
    path('exams/<uuid:exam_id>/preview/', views.exam_preview_view, name='exam_preview'),
    path('exams/<uuid:exam_id>/release-results/', views.exam_release_results_view, name='exam_release_results'),
    path('exams/<uuid:exam_id>/assign-users/', views.exam_assign_users_view, name='exam_assign_users'),

    # ─── Questions (AJAX) ───────────────────────────
    path('exams/<uuid:exam_id>/questions/add/', views.question_add_api, name='question_add'),
    path('questions/<uuid:question_id>/edit/', views.question_edit_api, name='question_edit'),
    path('questions/<uuid:question_id>/delete/', views.question_delete_api, name='question_delete'),

    # ─── Exam Taking ────────────────────────────────
    path('exam/<uuid:exam_id>/start/', views.start_exam_view, name='start_exam'),
    path('attempt/<uuid:attempt_id>/take/', views.take_exam_view, name='take_exam'),
    path('attempt/<uuid:attempt_id>/save-answer/', views.save_answer_api, name='save_answer'),
    path('attempt/<uuid:attempt_id>/submit/', views.submit_exam_api, name='submit_exam'),
    path('attempt/<uuid:attempt_id>/result/', views.exam_result_view, name='exam_result'),

    # ─── Results ────────────────────────────────────
    path('results/', views.results_dashboard_view, name='results_dashboard'),
    path('results/export/<uuid:exam_id>/', views.export_results_view, name='export_results'),

    # ─── Users ──────────────────────────────────────
    path('users/', views.users_list_view, name='users_list'),
]