# designer/views/exam_views.py

import json
import random
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.db import transaction

from ..models import Document, Exam, Question, GradeRule
from ..forms import ExamForm, QuestionForm, GenerateQuestionsForm
from ..services.pdf_service import PDFService
from ..services.question_generator import QuestionGenerator


@login_required
def generate_exam_view(request):
    """Page for selecting documents and generating questions."""
    if not request.user.is_staff:
        return redirect('designer:user_dashboard')

    documents = Document.objects.filter(uploaded_by=request.user, status='ready')
    context = {
        'documents': documents,
    }
    return render(request, 'designer/generate_exam.html', context)


@login_required
@require_POST
def generate_questions_api(request):
    """AJAX endpoint to generate questions from selected documents."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid request data.'})

    doc_ids = data.get('documents', [])
    source_type = data.get('source_type', 'entire')
    page_from = data.get('page_from')
    page_to = data.get('page_to')
    specific_pages = data.get('specific_pages', '')
    random_page_count = data.get('random_page_count')
    question_count_choice = data.get('question_count_choice', '10')
    custom_count = data.get('custom_count')

    if not doc_ids:
        return JsonResponse({'success': False, 'error': 'No documents selected.'})

    # Determine number of questions
    if question_count_choice == 'custom':
        try:
            num_questions = int(custom_count)
            if num_questions < 1 or num_questions > 500:
                return JsonResponse({'success': False, 'error': 'Custom count must be between 1 and 500.'})
        except (TypeError, ValueError):
            return JsonResponse({'success': False, 'error': 'Invalid custom question count.'})
    else:
        try:
            num_questions = int(question_count_choice)
        except ValueError:
            num_questions = 10

    # Collect text from all selected documents
    all_text_by_page = {}  # {(doc_id, page_num): text}
    doc_map = {}  # doc_id -> document

    documents = Document.objects.filter(
        id__in=doc_ids, uploaded_by=request.user, status='ready'
    )

    if not documents.exists():
        return JsonResponse({'success': False, 'error': 'No valid documents found.'})

    for doc in documents:
        doc_map[str(doc.id)] = doc
        file_path = doc.file.path

        # Determine which pages to extract
        if source_type == 'entire':
            pages = None  # All pages
        elif source_type == 'range':
            try:
                p_from = int(page_from) if page_from else 1
                p_to = int(page_to) if page_to else doc.page_count
                pages = list(range(p_from, p_to + 1))
            except (TypeError, ValueError):
                pages = None
        elif source_type == 'specific':
            pages = PDFService.parse_specific_pages(specific_pages or '')
            if not pages:
                return JsonResponse({'success': False, 'error': 'No valid page numbers specified.'})
        elif source_type == 'random':
            try:
                count = int(random_page_count) if random_page_count else 5
                pages = PDFService.get_random_pages(file_path, count)
            except (TypeError, ValueError):
                pages = PDFService.get_random_pages(file_path, 5)
        else:
            pages = None

        extracted = PDFService.extract_text_from_pages(file_path, pages)
        for page_num, text in extracted.items():
            all_text_by_page[(str(doc.id), page_num)] = text

    if not all_text_by_page:
        return JsonResponse({
            'success': False,
            'error': 'Could not extract text from the selected pages. The PDF may be scanned or empty.'
        })

    # Flatten to simple page->text dict for generator, keeping track of source
    simple_text = {}
    page_to_doc = {}  # page_key -> doc_id
    counter = 1
    for (doc_id, page_num), text in all_text_by_page.items():
        # Use a unique key to avoid collisions between documents
        key = counter
        simple_text[key] = f"[Source: {doc_map[doc_id].title}, Page {page_num}]\n{text}"
        page_to_doc[key] = (doc_id, page_num)
        counter += 1

    # Generate questions
    generator = QuestionGenerator()
    raw_questions = generator.generate_questions(simple_text, num_questions)

    if not raw_questions:
        return JsonResponse({
            'success': False,
            'error': 'Could not generate questions. Try selecting more content or fewer questions.'
        })

    # Map source pages back to documents
    page_keys = list(page_to_doc.keys())
    for q in raw_questions:
        source_page = q.get('source_page')
        if source_page and source_page in page_to_doc:
            doc_id, actual_page = page_to_doc[source_page]
            q['source_document_id'] = doc_id
            q['source_page'] = actual_page
        elif page_keys:
            # Default to first page
            doc_id, actual_page = page_to_doc[page_keys[0]]
            q['source_document_id'] = doc_id
            q['source_page'] = actual_page
        else:
            q['source_document_id'] = None
            q['source_page'] = None

    # Store in session for review before saving
    request.session['generated_questions'] = raw_questions
    request.session['generation_doc_ids'] = doc_ids

    return JsonResponse({
        'success': True,
        'questions': raw_questions,
        'count': len(raw_questions),
    })


@login_required
def review_questions_view(request):
    """Review generated questions before creating an exam."""
    if not request.user.is_staff:
        return redirect('designer:user_dashboard')

    questions = request.session.get('generated_questions', [])
    doc_ids = request.session.get('generation_doc_ids', [])

    if not questions:
        return redirect('designer:generate_exam')

    documents = Document.objects.filter(id__in=doc_ids)
    users = User.objects.filter(is_staff=False, is_active=True)

    context = {
        'questions': questions,
        'documents': documents,
        'users': users,
        'exam_form': ExamForm(),
    }
    return render(request, 'designer/review_questions.html', context)


@login_required
@require_POST
def save_exam_view(request):
    """Save the exam with generated questions."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid data.'})

    exam_data = data.get('exam', {})
    questions_data = data.get('questions', [])
    grade_rules_data = data.get('grade_rules', [])
    assigned_user_ids = data.get('assigned_users', [])
    save_as = data.get('save_as', 'draft')  # 'draft' or 'published'

    if not questions_data:
        return JsonResponse({'success': False, 'error': 'No questions to save.'})

    try:
        with transaction.atomic():
            # Create exam
            exam = Exam(
                name=exam_data.get('name', 'Untitled Exam'),
                description=exam_data.get('description', ''),
                institution=exam_data.get('institution', ''),
                course=exam_data.get('course', ''),
                department=exam_data.get('department', ''),
                exam_code=exam_data.get('exam_code', ''),
                instructor=exam_data.get('instructor', ''),
                instructions=exam_data.get('instructions', ''),
                answer_mode=exam_data.get('answer_mode', 'exam'),
                result_timing=exam_data.get('result_timing', 'immediate'),
                assignment_type=exam_data.get('assignment_type', 'all'),
                duration_minutes=int(exam_data.get('duration_minutes', 60)),
                max_attempts=int(exam_data.get('max_attempts', 1)),
                pass_mark=int(exam_data.get('pass_mark', 50)),
                randomize_questions=exam_data.get('randomize_questions', False),
                randomize_options=exam_data.get('randomize_options', False),
                pass_message=exam_data.get('pass_message', 'Congratulations! You passed the examination.'),
                fail_message=exam_data.get('fail_message', 'Unfortunately, you did not meet the required pass mark.'),
                status=save_as,
                created_by=request.user,
            )

            # Handle dates
            scheduled_date = exam_data.get('scheduled_date')
            if scheduled_date:
                exam.scheduled_date = scheduled_date
            start_time = exam_data.get('start_time')
            if start_time:
                exam.start_time = start_time
            end_time = exam_data.get('end_time')
            if end_time:
                exam.end_time = end_time

            exam.save()

            # Add source documents
            doc_ids = request.session.get('generation_doc_ids', [])
            if doc_ids:
                docs = Document.objects.filter(id__in=doc_ids)
                exam.source_documents.set(docs)

            # Assign users
            if exam.assignment_type == 'specific' and assigned_user_ids:
                users = User.objects.filter(id__in=assigned_user_ids)
                exam.assigned_users.set(users)

            # Save questions
            for idx, q in enumerate(questions_data):
                source_doc = None
                source_doc_id = q.get('source_document_id')
                if source_doc_id:
                    try:
                        source_doc = Document.objects.get(id=source_doc_id)
                    except Document.DoesNotExist:
                        pass

                Question.objects.create(
                    exam=exam,
                    question_text=q.get('question_text', ''),
                    option_a=q.get('option_a', ''),
                    option_b=q.get('option_b', ''),
                    option_c=q.get('option_c', ''),
                    option_d=q.get('option_d', ''),
                    correct_answer=q.get('correct_answer', 'A'),
                    explanation=q.get('explanation', ''),
                    source_document=source_doc,
                    source_page=q.get('source_page'),
                    order=idx + 1,
                )

            # Save grade rules
            if grade_rules_data:
                for rule in grade_rules_data:
                    GradeRule.objects.create(
                        exam=exam,
                        grade=rule.get('grade', ''),
                        min_score=int(rule.get('min_score', 0)),
                        max_score=int(rule.get('max_score', 100)),
                    )
            else:
                # Default grading
                default_grades = [
                    ('A', 70, 100),
                    ('B', 60, 69),
                    ('C', 50, 59),
                    ('D', 45, 49),
                    ('E', 40, 44),
                    ('F', 0, 39),
                ]
                for grade, min_s, max_s in default_grades:
                    GradeRule.objects.create(
                        exam=exam, grade=grade, min_score=min_s, max_score=max_s
                    )

            # Clear session data
            request.session.pop('generated_questions', None)
            request.session.pop('generation_doc_ids', None)

        return JsonResponse({
            'success': True,
            'exam_id': str(exam.id),
            'message': f'Exam "{exam.name}" saved as {save_as}.',
            'redirect': f'/exams/{exam.id}/'
        })

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
def exam_list_view(request):
    """List all exams (admin)."""
    if not request.user.is_staff:
        return redirect('designer:user_dashboard')

    exams = Exam.objects.filter(created_by=request.user)
    context = {'exams': exams}
    return render(request, 'designer/exam_list.html', context)


@login_required
def exam_detail_view(request, exam_id):
    """View exam details."""
    if not request.user.is_staff:
        return redirect('designer:user_dashboard')

    exam = get_object_or_404(Exam, id=exam_id, created_by=request.user)
    questions = exam.questions.all()
    grade_rules = exam.grade_rules.all()
    attempts = exam.attempts.select_related('user').all()

    context = {
        'exam': exam,
        'questions': questions,
        'grade_rules': grade_rules,
        'attempts': attempts,
    }
    return render(request, 'designer/exam_detail.html', context)


@login_required
def exam_edit_view(request, exam_id):
    """Edit an exam."""
    if not request.user.is_staff:
        return redirect('designer:user_dashboard')

    exam = get_object_or_404(Exam, id=exam_id, created_by=request.user)

    if request.method == 'POST':
        form = ExamForm(request.POST, instance=exam)
        if form.is_valid():
            form.save()
            return redirect('designer:exam_detail', exam_id=exam.id)
    else:
        form = ExamForm(instance=exam)

    questions = exam.questions.all()
    users = User.objects.filter(is_staff=False, is_active=True)

    context = {
        'exam': exam,
        'form': form,
        'questions': questions,
        'users': users,
    }
    return render(request, 'designer/exam_edit.html', context)


@login_required
@require_POST
def exam_publish_view(request, exam_id):
    """Publish/unpublish an exam (AJAX)."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    exam = get_object_or_404(Exam, id=exam_id, created_by=request.user)

    if exam.status in ['draft', 'unpublished']:
        if exam.questions.count() == 0:
            return JsonResponse({'success': False, 'error': 'Cannot publish an exam with no questions.'})
        exam.status = 'published'
        exam.save()
        return JsonResponse({'success': True, 'status': 'published', 'message': 'Exam published.'})
    elif exam.status == 'published':
        exam.status = 'unpublished'
        exam.save()
        return JsonResponse({'success': True, 'status': 'unpublished', 'message': 'Exam unpublished.'})

    return JsonResponse({'success': False, 'error': 'Invalid status transition.'})


@login_required
@require_POST
def exam_delete_view(request, exam_id):
    """Delete an exam (AJAX)."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    exam = get_object_or_404(Exam, id=exam_id, created_by=request.user)
    name = exam.name
    exam.delete()
    return JsonResponse({'success': True, 'message': f'Exam "{name}" deleted.'})


@login_required
@require_POST
def exam_duplicate_view(request, exam_id):
    """Duplicate an exam (AJAX)."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    original = get_object_or_404(Exam, id=exam_id, created_by=request.user)

    with transaction.atomic():
        # Create exam copy
        new_exam = Exam.objects.create(
            name=f"{original.name} (Copy)",
            description=original.description,
            institution=original.institution,
            course=original.course,
            department=original.department,
            exam_code=f"{original.exam_code}-COPY" if original.exam_code else '',
            instructor=original.instructor,
            instructions=original.instructions,
            answer_mode=original.answer_mode,
            result_timing=original.result_timing,
            assignment_type=original.assignment_type,
            duration_minutes=original.duration_minutes,
            max_attempts=original.max_attempts,
            pass_mark=original.pass_mark,
            randomize_questions=original.randomize_questions,
            randomize_options=original.randomize_options,
            pass_message=original.pass_message,
            fail_message=original.fail_message,
            status='draft',
            created_by=request.user,
        )

        # Copy source documents
        new_exam.source_documents.set(original.source_documents.all())

        # Copy questions
        for q in original.questions.all():
            Question.objects.create(
                exam=new_exam,
                question_text=q.question_text,
                option_a=q.option_a,
                option_b=q.option_b,
                option_c=q.option_c,
                option_d=q.option_d,
                correct_answer=q.correct_answer,
                explanation=q.explanation,
                source_document=q.source_document,
                source_page=q.source_page,
                order=q.order,
            )

        # Copy grade rules
        for rule in original.grade_rules.all():
            GradeRule.objects.create(
                exam=new_exam,
                grade=rule.grade,
                min_score=rule.min_score,
                max_score=rule.max_score,
            )

    return JsonResponse({
        'success': True,
        'exam_id': str(new_exam.id),
        'message': f'Exam duplicated as "{new_exam.name}".',
    })


@login_required
def exam_preview_view(request, exam_id):
    """Preview an exam as a student would see it."""
    if not request.user.is_staff:
        return redirect('designer:user_dashboard')

    exam = get_object_or_404(Exam, id=exam_id, created_by=request.user)
    questions = list(exam.questions.all())

    context = {
        'exam': exam,
        'questions': questions,
        'is_preview': True,
    }
    return render(request, 'designer/exam_preview.html', context)


@login_required
@require_POST
def question_edit_api(request, question_id):
    """Edit a question (AJAX)."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    question = get_object_or_404(Question, id=question_id, exam__created_by=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid data.'})

    question.question_text = data.get('question_text', question.question_text)
    question.option_a = data.get('option_a', question.option_a)
    question.option_b = data.get('option_b', question.option_b)
    question.option_c = data.get('option_c', question.option_c)
    question.option_d = data.get('option_d', question.option_d)
    correct = data.get('correct_answer', question.correct_answer)
    if correct in ['A', 'B', 'C', 'D']:
        question.correct_answer = correct
    question.explanation = data.get('explanation', question.explanation)
    question.save()

    return JsonResponse({'success': True, 'message': 'Question updated.'})


@login_required
@require_POST
def question_delete_api(request, question_id):
    """Delete a question (AJAX)."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    question = get_object_or_404(Question, id=question_id, exam__created_by=request.user)
    question.delete()
    return JsonResponse({'success': True, 'message': 'Question deleted.'})


@login_required
@require_POST
def question_add_api(request, exam_id):
    """Add a question manually (AJAX)."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    exam = get_object_or_404(Exam, id=exam_id, created_by=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid data.'})

    max_order = exam.questions.count()

    question = Question.objects.create(
        exam=exam,
        question_text=data.get('question_text', 'New question'),
        option_a=data.get('option_a', 'Option A'),
        option_b=data.get('option_b', 'Option B'),
        option_c=data.get('option_c', 'Option C'),
        option_d=data.get('option_d', 'Option D'),
        correct_answer=data.get('correct_answer', 'A'),
        explanation=data.get('explanation', ''),
        order=max_order + 1,
    )

    return JsonResponse({
        'success': True,
        'question_id': str(question.id),
        'message': 'Question added.',
    })


@login_required
@require_POST
def exam_release_results_view(request, exam_id):
    """Release results for an exam (AJAX)."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    exam = get_object_or_404(Exam, id=exam_id, created_by=request.user)
    exam.results_released = True
    exam.save()
    return JsonResponse({'success': True, 'message': 'Results released.'})


@login_required
@require_POST
def exam_assign_users_view(request, exam_id):
    """Assign users to an exam (AJAX)."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    exam = get_object_or_404(Exam, id=exam_id, created_by=request.user)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid data.'})

    assignment_type = data.get('assignment_type', exam.assignment_type)
    user_ids = data.get('user_ids', [])

    exam.assignment_type = assignment_type
    exam.save()

    if assignment_type == 'specific':
        users = User.objects.filter(id__in=user_ids)
        exam.assigned_users.set(users)
    else:
        exam.assigned_users.clear()

    return JsonResponse({'success': True, 'message': 'Assignment updated.'})