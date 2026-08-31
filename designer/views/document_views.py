# designer/views/document_views.py

import os
import uuid
import json
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.db.models import Sum
from django.conf import settings

from ..models import Document
from ..services.pdf_service import PDFService
from ..services.document_converter import DocumentConverter

MAX_TOTAL_UPLOAD_SIZE = 1 * 1024 * 1024 * 1024  # 1 GB
ALLOWED_EXTENSIONS = ['pdf', 'docx', 'doc', 'pptx', 'ppt', 'odt', 'txt']


@login_required
def documents_view(request):
    """Document library page."""
    if not request.user.is_staff:
        return redirect('designer:user_dashboard')

    documents = Document.objects.filter(uploaded_by=request.user)
    total_size = documents.aggregate(total=Sum('file_size'))['total'] or 0
    remaining_size = MAX_TOTAL_UPLOAD_SIZE - total_size

    context = {
        'documents': documents,
        'total_size': total_size,
        'total_size_display': _format_size(total_size),
        'remaining_size': remaining_size,
        'remaining_size_display': _format_size(remaining_size),
        'max_size_display': _format_size(MAX_TOTAL_UPLOAD_SIZE),
        'usage_percent': round((total_size / MAX_TOTAL_UPLOAD_SIZE) * 100, 1) if MAX_TOTAL_UPLOAD_SIZE else 0,
    }
    return render(request, 'designer/documents.html', context)


@login_required
@require_POST
def document_upload_view(request):
    """Handle file upload (AJAX)."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    files = request.FILES.getlist('files')
    if not files:
        return JsonResponse({'success': False, 'error': 'No files selected.'})

    # Calculate current total size
    current_total = Document.objects.filter(
        uploaded_by=request.user
    ).aggregate(total=Sum('file_size'))['total'] or 0

    # Calculate new files total size
    new_files_size = sum(f.size for f in files)

    if current_total + new_files_size > MAX_TOTAL_UPLOAD_SIZE:
        return JsonResponse({
            'success': False,
            'error': f'Upload would exceed 1 GB limit. Current usage: {_format_size(current_total)}. '
                     f'Remaining: {_format_size(MAX_TOTAL_UPLOAD_SIZE - current_total)}. '
                     f'Attempted: {_format_size(new_files_size)}.'
        })

    uploaded = []
    errors = []

    for f in files:
        filename = f.name
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

        if ext not in ALLOWED_EXTENSIONS:
            errors.append(f"'{filename}': Unsupported file type '.{ext}'")
            continue

        # Check for duplicate filename
        if Document.objects.filter(uploaded_by=request.user, original_filename=filename).exists():
            errors.append(f"'{filename}': A file with this name already exists.")
            continue

        try:
            import tempfile
            
            # Save uploaded file to temp location for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as temp_file:
                for chunk in f.chunks():
                    temp_file.write(chunk)
                temp_path = temp_file.name

            try:
                # Convert non-PDF files
                if ext != 'pdf':
                    converted_path = DocumentConverter.convert_to_pdf(temp_path, filename)
                    if converted_path and converted_path != temp_path:
                        # Use the converted PDF
                        with open(converted_path, 'rb') as converted_file:
                            from django.core.files import File
                            doc = Document(
                                title=filename.rsplit('.', 1)[0] if '.' in filename else filename,
                                original_filename=filename,
                                file=File(converted_file, name=filename.rsplit('.', 1)[0] + '.pdf'),
                                file_type='pdf',
                                file_size=os.path.getsize(converted_path),
                                status='processing',
                                uploaded_by=request.user,
                            )
                            doc.save()
                        # Clean up converted file
                        try:
                            os.remove(converted_path)
                        except OSError:
                            pass
                    elif converted_path is None:
                        # Store original file if conversion failed
                        with open(temp_path, 'rb') as original_file:
                            from django.core.files import File
                            doc = Document(
                                title=filename.rsplit('.', 1)[0] if '.' in filename else filename,
                                original_filename=filename,
                                file=File(original_file, name=filename),
                                file_type=ext,
                                file_size=f.size,
                                status='error',
                                error_message=f'Could not convert .{ext} to PDF. Stored original file.',
                                uploaded_by=request.user,
                            )
                            doc.save()
                        uploaded.append({
                            'id': str(doc.id),
                            'name': doc.original_filename,
                            'size': doc.file_size_display,
                            'status': doc.status,
                        })
                        continue
                else:
                    # PDF file - save directly
                    with open(temp_path, 'rb') as pdf_file:
                        from django.core.files import File
                        doc = Document(
                            title=filename.rsplit('.', 1)[0] if '.' in filename else filename,
                            original_filename=filename,
                            file=File(pdf_file, name=filename),
                            file_type='pdf',
                            file_size=f.size,
                            status='processing',
                            uploaded_by=request.user,
                        )
                        doc.save()

                # Get page count for PDFs
                try:
                    if doc.file_type == 'pdf':
                        doc.page_count = PDFService.get_page_count(temp_path)
                    doc.status = 'ready'
                except Exception as e:
                    doc.status = 'error'
                    doc.error_message = str(e)

                doc.save()
                uploaded.append({
                    'id': str(doc.id),
                    'name': doc.original_filename,
                    'size': doc.file_size_display,
                    'pages': doc.page_count,
                    'status': doc.status,
                })

            finally:
                # Clean up temp file
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

        except Exception as e:
            errors.append(f"'{filename}': {str(e)}")

    return JsonResponse({
        'success': True,
        'uploaded': uploaded,
        'errors': errors,
        'message': f'{len(uploaded)} file(s) uploaded successfully.' + (
            f' {len(errors)} error(s).' if errors else ''
        )
    })


@login_required
@require_POST
def document_delete_view(request, doc_id):
    """Delete a single document (AJAX)."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    doc = get_object_or_404(Document, id=doc_id, uploaded_by=request.user)

    # Delete the actual file
    try:
        if doc.file and os.path.isfile(doc.file.path):
            os.remove(doc.file.path)
    except Exception:
        pass

    doc.delete()
    return JsonResponse({'success': True, 'message': 'Document deleted.'})


@login_required
@require_POST
def documents_reset_view(request):
    """Delete all documents for the current user (AJAX)."""
    if not request.user.is_staff:
        return JsonResponse({'success': False, 'error': 'Permission denied.'}, status=403)

    documents = Document.objects.filter(uploaded_by=request.user)

    for doc in documents:
        try:
            if doc.file and os.path.isfile(doc.file.path):
                os.remove(doc.file.path)
        except Exception:
            pass

    count = documents.count()
    documents.delete()

    return JsonResponse({
        'success': True,
        'message': f'{count} document(s) removed.'
    })


def _format_size(size_bytes):
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"