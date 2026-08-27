# designer/views/__init__.py

# Home
from .home_views import home

# Auth & Dashboards
from .auth_views import (
    admin_login_view,
    user_login_view,
    user_signup_view,
    logout_view,
    admin_dashboard_view,
    user_dashboard_view,
)

# Documents
from .document_views import (
    documents_view,
    document_upload_view,
    document_delete_view,
    documents_reset_view,
)

# Exams
from .exam_views import (
    generate_exam_view,
    generate_start_api,      # <-- Added
    generate_batch_api,      # <-- Added
    generate_quota_api,      # <-- Added
    review_questions_view,
    save_exam_view,
    exam_list_view,
    exam_detail_view,
    exam_edit_view,
    exam_publish_view,
    exam_delete_view,
    exam_duplicate_view,
    exam_preview_view,
    exam_release_results_view,
    exam_assign_users_view,
    question_edit_api,
    question_delete_api,
    question_add_api,
)

# Attempts
from .attempt_views import (
    start_exam_view,
    take_exam_view,
    save_answer_api,
    submit_exam_api,
    exam_result_view,
)

# Results
from .result_views import (
    results_dashboard_view,
    export_results_view,
    users_list_view,
)