# designer/forms.py

from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from .models import Exam, Question, Document


# ─────────────────────────────────────────
# User Signup Form
# ─────────────────────────────────────────
class UserSignupForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={
            'placeholder': 'Email address',
            'class': 'form-input'
        })
    )
    first_name = forms.CharField(
        max_length=50,
        required=True,
        widget=forms.TextInput(attrs={
            'placeholder': 'First name',
            'class': 'form-input'
        })
    )
    last_name = forms.CharField(
        max_length=50,
        required=True,
        widget=forms.TextInput(attrs={
            'placeholder': 'Last name',
            'class': 'form-input'
        })
    )

    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email', 'username', 'password1', 'password2']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            field.widget.attrs.setdefault('class', 'form-input')
            field.widget.attrs.setdefault('placeholder', field.label)

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("An account with this email already exists.")
        return email


# ─────────────────────────────────────────
# Admin Login Form
# ─────────────────────────────────────────
class AdminLoginForm(forms.Form):
    username = forms.CharField(
        max_length=150,
        widget=forms.TextInput(attrs={
            'placeholder': 'Admin username',
            'class': 'form-input',
            'autofocus': True
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'placeholder': 'Password',
            'class': 'form-input'
        })
    )


# ─────────────────────────────────────────
# User Login Form
# ─────────────────────────────────────────
class UserLoginForm(forms.Form):
    username = forms.CharField(
        max_length=150,
        widget=forms.TextInput(attrs={
            'placeholder': 'Username or Email',
            'class': 'form-input',
            'autofocus': True
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'placeholder': 'Password',
            'class': 'form-input'
        })
    )
    remember_me = forms.BooleanField(
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-checkbox'
        })
    )


# ─────────────────────────────────────────
# PDF Upload Form
# ─────────────────────────────────────────
class PDFUploadForm(forms.Form):
    title = forms.CharField(
        max_length=500,
        required=False,
        widget=forms.TextInput(attrs={
            'placeholder': 'Exam title (optional)',
            'class': 'form-input'
        })
    )
    pdf_file = forms.FileField(
        widget=forms.FileInput(attrs={
            'accept': '.pdf,.docx,.doc,.pptx,.ppt,.odt,.txt',
            'class': 'form-file'
        })
    )
    num_questions = forms.IntegerField(
        min_value=1,
        max_value=100,
        initial=10,
        widget=forms.NumberInput(attrs={
            'placeholder': 'Number of questions',
            'class': 'form-input'
        })
    )
    difficulty = forms.ChoiceField(
        choices=[
            ('easy', 'Easy'),
            ('medium', 'Medium'),
            ('hard', 'Hard'),
            ('mixed', 'Mixed'),
        ],
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    duration_minutes = forms.IntegerField(
        min_value=1,
        max_value=300,
        initial=60,
        widget=forms.NumberInput(attrs={
            'placeholder': 'Duration in minutes',
            'class': 'form-input'
        })
    )
    pass_mark = forms.IntegerField(
        min_value=0,
        max_value=100,
        initial=50,
        widget=forms.NumberInput(attrs={
            'placeholder': 'Pass mark (%)',
            'class': 'form-input'
        })
    )

    def clean_pdf_file(self):
        pdf = self.cleaned_data.get('pdf_file')
        if pdf:
            allowed = ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.odt', '.txt']
            ext = '.' + pdf.name.split('.')[-1].lower()
            if ext not in allowed:
                raise forms.ValidationError("Unsupported file type.")
            if pdf.size > 20 * 1024 * 1024:
                raise forms.ValidationError("File size must be under 20MB.")
        return pdf


# ─────────────────────────────────────────
# Exam Form (matches Exam model exactly)
# ─────────────────────────────────────────
class ExamForm(forms.ModelForm):
    class Meta:
        model = Exam
        fields = [
            'name',
            'description',
            'institution',
            'course',
            'department',
            'exam_code',
            'instructor',
            'instructions',
            'status',
            'answer_mode',
            'result_timing',
            'assignment_type',
            'duration_minutes',
            'max_attempts',
            'pass_mark',
            'randomize_questions',
            'randomize_options',
            'scheduled_date',
            'start_time',
            'end_time',
            'pass_message',
            'fail_message',
        ]
        widgets = {
            # Basic Info
            'name': forms.TextInput(attrs={
                'placeholder': 'Exam name',
                'class': 'form-input'
            }),
            'description': forms.Textarea(attrs={
                'placeholder': 'Exam description (optional)',
                'class': 'form-input',
                'rows': 3
            }),
            'institution': forms.TextInput(attrs={
                'placeholder': 'Institution name',
                'class': 'form-input'
            }),
            'course': forms.TextInput(attrs={
                'placeholder': 'Course name',
                'class': 'form-input'
            }),
            'department': forms.TextInput(attrs={
                'placeholder': 'Department',
                'class': 'form-input'
            }),
            'exam_code': forms.TextInput(attrs={
                'placeholder': 'Exam code (e.g. EE301)',
                'class': 'form-input'
            }),
            'instructor': forms.TextInput(attrs={
                'placeholder': 'Instructor name',
                'class': 'form-input'
            }),
            'instructions': forms.Textarea(attrs={
                'placeholder': 'Exam instructions for students...',
                'class': 'form-input',
                'rows': 4
            }),
            # Settings
            'status': forms.Select(attrs={'class': 'form-select'}),
            'answer_mode': forms.Select(attrs={'class': 'form-select'}),
            'result_timing': forms.Select(attrs={'class': 'form-select'}),
            'assignment_type': forms.Select(attrs={'class': 'form-select'}),
            # Numbers
            'duration_minutes': forms.NumberInput(attrs={
                'placeholder': 'Duration in minutes',
                'class': 'form-input'
            }),
            'max_attempts': forms.NumberInput(attrs={
                'placeholder': 'Max attempts allowed',
                'class': 'form-input'
            }),
            'pass_mark': forms.NumberInput(attrs={
                'placeholder': 'Pass mark (%)',
                'class': 'form-input'
            }),
            # Booleans
            'randomize_questions': forms.CheckboxInput(attrs={
                'class': 'form-checkbox'
            }),
            'randomize_options': forms.CheckboxInput(attrs={
                'class': 'form-checkbox'
            }),
            # Schedule
            'scheduled_date': forms.DateInput(attrs={
                'type': 'date',
                'class': 'form-input'
            }),
            'start_time': forms.TimeInput(attrs={
                'type': 'time',
                'class': 'form-input'
            }),
            'end_time': forms.TimeInput(attrs={
                'type': 'time',
                'class': 'form-input'
            }),
            # Messages
            'pass_message': forms.Textarea(attrs={
                'class': 'form-input',
                'rows': 2
            }),
            'fail_message': forms.Textarea(attrs={
                'class': 'form-input',
                'rows': 2
            }),
        }


# ─────────────────────────────────────────
# Question Form (matches Question model)
# ─────────────────────────────────────────
class QuestionForm(forms.ModelForm):
    class Meta:
        model = Question
        fields = [
            'question_text',
            'option_a',
            'option_b',
            'option_c',
            'option_d',
            'correct_answer',
            'explanation',
            'order',
        ]
        widgets = {
            'question_text': forms.Textarea(attrs={
                'placeholder': 'Enter question text...',
                'class': 'form-input',
                'rows': 3
            }),
            'option_a': forms.Textarea(attrs={
                'placeholder': 'Option A',
                'class': 'form-input',
                'rows': 2
            }),
            'option_b': forms.Textarea(attrs={
                'placeholder': 'Option B',
                'class': 'form-input',
                'rows': 2
            }),
            'option_c': forms.Textarea(attrs={
                'placeholder': 'Option C',
                'class': 'form-input',
                'rows': 2
            }),
            'option_d': forms.Textarea(attrs={
                'placeholder': 'Option D',
                'class': 'form-input',
                'rows': 2
            }),
            'correct_answer': forms.Select(attrs={
                'class': 'form-select'
            }),
            'explanation': forms.Textarea(attrs={
                'placeholder': 'Explanation (optional)',
                'class': 'form-input',
                'rows': 2
            }),
            'order': forms.NumberInput(attrs={
                'class': 'form-input'
            }),
        }


# ─────────────────────────────────────────
# Generate Questions Form
# ─────────────────────────────────────────
class GenerateQuestionsForm(forms.Form):
    num_questions = forms.IntegerField(
        min_value=1,
        max_value=100,
        initial=10,
        label='Number of Questions',
        widget=forms.NumberInput(attrs={
            'placeholder': 'Number of questions',
            'class': 'form-input'
        })
    )
    difficulty = forms.ChoiceField(
        choices=[
            ('easy', 'Easy'),
            ('medium', 'Medium'),
            ('hard', 'Hard'),
            ('mixed', 'Mixed'),
        ],
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    focus_topic = forms.CharField(
        required=False,
        max_length=200,
        label='Focus Topic (optional)',
        widget=forms.TextInput(attrs={
            'placeholder': "e.g. Chapter 3, Ohm's Law...",
            'class': 'form-input'
        })
    )