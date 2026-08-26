# designer/admin.py

from django.contrib import admin
from .models import Document, Exam, Question, GradeRule, ExamAttempt, UserAnswer


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ['title', 'original_filename', 'file_type', 'file_size', 'page_count', 'status', 'uploaded_by', 'uploaded_at']
    list_filter = ['status', 'file_type']
    search_fields = ['title', 'original_filename']


class QuestionInline(admin.TabularInline):
    model = Question
    extra = 0
    fields = ['order', 'question_text', 'correct_answer']


class GradeRuleInline(admin.TabularInline):
    model = GradeRule
    extra = 0


@admin.register(Exam)
class ExamAdmin(admin.ModelAdmin):
    list_display = ['name', 'status', 'answer_mode', 'question_count', 'created_by', 'created_at']
    list_filter = ['status', 'answer_mode']
    search_fields = ['name', 'course']
    inlines = [QuestionInline, GradeRuleInline]


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ['exam', 'order', 'question_text', 'correct_answer']
    list_filter = ['exam']


@admin.register(ExamAttempt)
class ExamAttemptAdmin(admin.ModelAdmin):
    list_display = ['user', 'exam', 'status', 'score', 'percentage', 'grade', 'passed', 'start_time']
    list_filter = ['status', 'passed']
    search_fields = ['user__username']


@admin.register(UserAnswer)
class UserAnswerAdmin(admin.ModelAdmin):
    list_display = ['attempt', 'question', 'selected_answer', 'is_correct']