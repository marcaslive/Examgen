# designer/models.py

import uuid
from django.db import models
from django.conf import settings
from django.utils import timezone
from django.core.validators import MinValueValidator, MaxValueValidator, FileExtensionValidator


def document_upload_path(instance, filename):
    return f'documents/{instance.uploaded_by.id}/{filename}'


class Document(models.Model):
    STATUS_CHOICES = [
        ('uploaded', 'Uploaded'),
        ('processing', 'Processing'),
        ('ready', 'Ready'),
        ('error', 'Error'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=500)
    original_filename = models.CharField(max_length=500)
    file = models.FileField(
        upload_to=document_upload_path,
        validators=[FileExtensionValidator(
            allowed_extensions=['pdf', 'docx', 'doc', 'pptx', 'ppt', 'odt', 'txt']
        )]
    )
    file_type = models.CharField(max_length=20, default='pdf')
    file_size = models.BigIntegerField(default=0)  # bytes
    page_count = models.IntegerField(default=0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='uploaded')
    error_message = models.TextField(blank=True, default='')
    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='documents'
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-uploaded_at']
        indexes = [
            models.Index(fields=['uploaded_by', '-uploaded_at']),
            models.Index(fields=['status']),
        ]

    def __str__(self):
        return self.title

    @property
    def file_size_display(self):
        size = self.file_size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"


class Exam(models.Model):
    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('published', 'Published'),
        ('unpublished', 'Unpublished'),
        ('archived', 'Archived'),
    ]

    ANSWER_MODE_CHOICES = [
        ('study', 'Questions + Answers (Study Mode)'),
        ('exam', 'Examination Mode'),
    ]

    RESULT_TIMING_CHOICES = [
        ('immediate', 'Show Score Immediately'),
        ('later', 'Show Score Later'),
    ]

    ASSIGN_CHOICES = [
        ('self', 'Self Only'),
        ('specific', 'Specific Users'),
        ('all', 'All Users'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=500)
    description = models.TextField(blank=True, default='')
    institution = models.CharField(max_length=300, blank=True, default='')
    course = models.CharField(max_length=300, blank=True, default='')
    department = models.CharField(max_length=300, blank=True, default='')
    exam_code = models.CharField(max_length=50, blank=True, default='')
    instructor = models.CharField(max_length=300, blank=True, default='')
    instructions = models.TextField(blank=True, default='')

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='draft')
    answer_mode = models.CharField(max_length=10, choices=ANSWER_MODE_CHOICES, default='exam')
    result_timing = models.CharField(max_length=10, choices=RESULT_TIMING_CHOICES, default='immediate')
    assignment_type = models.CharField(max_length=10, choices=ASSIGN_CHOICES, default='all')

    duration_minutes = models.PositiveIntegerField(default=60, validators=[MinValueValidator(1)])
    max_attempts = models.PositiveIntegerField(default=1, validators=[MinValueValidator(1)])
    pass_mark = models.PositiveIntegerField(default=50, validators=[MinValueValidator(0), MaxValueValidator(100)])

    randomize_questions = models.BooleanField(default=False)
    randomize_options = models.BooleanField(default=False)

    scheduled_date = models.DateField(null=True, blank=True)
    start_time = models.TimeField(null=True, blank=True)
    end_time = models.TimeField(null=True, blank=True)

    results_released = models.BooleanField(default=False)

    pass_message = models.TextField(default='Congratulations! You passed the examination.')
    fail_message = models.TextField(default='Unfortunately, you did not meet the required pass mark.')

    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='created_exams'
    )
    assigned_users = models.ManyToManyField(
        settings.AUTH_USER_MODEL, blank=True, related_name='assigned_exams'
    )
    source_documents = models.ManyToManyField(Document, blank=True, related_name='exams')

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status']),
            models.Index(fields=['created_by']),
            models.Index(fields=['-created_at']),
        ]

    def __str__(self):
        return self.name

    @property
    def question_count(self):
        return self.questions.count()

    @property
    def is_published(self):
        return self.status == 'published'


class GradeRule(models.Model):
    exam = models.ForeignKey(Exam, on_delete=models.CASCADE, related_name='grade_rules')
    grade = models.CharField(max_length=5)
    min_score = models.PositiveIntegerField()
    max_score = models.PositiveIntegerField()
    order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ['-min_score']
        unique_together = ['exam', 'grade']

    def __str__(self):
        return f"{self.grade}: {self.min_score}-{self.max_score}%"


class Question(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    exam = models.ForeignKey(Exam, on_delete=models.CASCADE, related_name='questions')
    question_text = models.TextField()
    option_a = models.TextField()
    option_b = models.TextField()
    option_c = models.TextField()
    option_d = models.TextField()
    correct_answer = models.CharField(max_length=1, choices=[
        ('A', 'A'), ('B', 'B'), ('C', 'C'), ('D', 'D')
    ])
    explanation = models.TextField(blank=True, default='')
    source_document = models.ForeignKey(
        Document, on_delete=models.SET_NULL, null=True, blank=True, related_name='questions'
    )
    source_page = models.PositiveIntegerField(null=True, blank=True)
    order = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['order', 'created_at']
        indexes = [
            models.Index(fields=['exam', 'order']),
        ]

    def __str__(self):
        return f"Q{self.order}: {self.question_text[:80]}"


class ExamAttempt(models.Model):
    STATUS_CHOICES = [
        ('in_progress', 'In Progress'),
        ('submitted', 'Submitted'),
        ('timed_out', 'Timed Out'),
        ('graded', 'Graded'),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='exam_attempts'
    )
    exam = models.ForeignKey(Exam, on_delete=models.CASCADE, related_name='attempts')
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)
    submitted_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='in_progress')
    score = models.PositiveIntegerField(default=0)
    total_questions = models.PositiveIntegerField(default=0)
    percentage = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    grade = models.CharField(max_length=5, blank=True, default='')
    passed = models.BooleanField(default=False)
    question_order = models.JSONField(default=list, blank=True)
    option_orders = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ['-start_time']
        indexes = [
            models.Index(fields=['user', 'exam']),
            models.Index(fields=['status']),
        ]

    def __str__(self):
        return f"{self.user.username} - {self.exam.name}"

    @property
    def time_spent_display(self):
        if self.submitted_at and self.start_time:
            delta = self.submitted_at - self.start_time
            total_seconds = int(delta.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            if hours:
                return f"{hours}h {minutes}m {seconds}s"
            return f"{minutes}m {seconds}s"
        return "N/A"

    @property
    def time_remaining_seconds(self):
        if self.status != 'in_progress':
            return 0
        elapsed = (timezone.now() - self.start_time).total_seconds()
        remaining = (self.exam.duration_minutes * 60) - elapsed
        return max(0, int(remaining))

    @property
    def is_expired(self):
        if self.status != 'in_progress':
            return False
        elapsed = (timezone.now() - self.start_time).total_seconds()
        return elapsed > (self.exam.duration_minutes * 60) + 30  # 30s grace


class UserAnswer(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    attempt = models.ForeignKey(ExamAttempt, on_delete=models.CASCADE, related_name='answers')
    question = models.ForeignKey(Question, on_delete=models.CASCADE, related_name='user_answers')
    selected_answer = models.CharField(max_length=1, blank=True, default='', choices=[
        ('', 'Not Answered'), ('A', 'A'), ('B', 'B'), ('C', 'C'), ('D', 'D')
    ])
    is_correct = models.BooleanField(default=False)
    answered_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ['attempt', 'question']
        indexes = [
            models.Index(fields=['attempt', 'question']),
        ]

    def __str__(self):
        return f"{self.attempt.user.username} - Q{self.question.order}: {self.selected_answer}"